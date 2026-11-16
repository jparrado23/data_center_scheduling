"""Reusable Pauli Correlation Encoding experiments for scheduling QUBOs.

The module deliberately separates PCE optimization from a particular quantum
backend.  Statevector and hardware callers provide a correlation evaluator:
given circuit parameters, it returns one expectation value per PCE Pauli
observable.  The shared decoder always validates the resulting schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize

from src.quantum.qubo_builder import (
    SchedulingQubo,
    decode_qubo_sample,
    qubo_energy,
    validate_decoded_schedule,
)


CorrelationEvaluator = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class IsingModel:
    """Ising coefficients using ``E(s) = c + h.s + sum(J_ij s_i s_j)``."""

    constant: float
    fields: np.ndarray
    left: np.ndarray
    right: np.ndarray
    couplings: np.ndarray


@dataclass(frozen=True)
class FixedAlphaConfig:
    """Configuration for one fixed-alpha PCE run.

    ``maxiter`` is passed as the objective-evaluation cap for the supported
    COBYLA and Powell optimizers.
    """

    maxiter: int = 120
    rhobeg: float = 0.5
    tolerance: float = 1e-5
    optimizer: str = "COBYLA"


@dataclass(frozen=True)
class ProgressiveBinarizationConfig:
    """Continuation parameters for PB-PCE."""

    initial_alpha: float = 3.0
    binarization_target: float = 0.90
    max_stages: int = 12
    alpha_max: float = 1e6
    minimum_alpha_growth: float = 1.01
    fixed_alpha: FixedAlphaConfig = FixedAlphaConfig()


@dataclass
class PceCandidate:
    """One continuous PCE state together with its decoded schedule."""

    theta: np.ndarray
    alpha: float
    correlations: np.ndarray
    soft_spins: np.ndarray
    bits: np.ndarray
    pce_loss: float
    qubo_energy: float
    schedule: pd.DataFrame
    validation: dict[str, Any]


@dataclass
class FixedAlphaRun:
    """Optimizer output and selected candidate for a single alpha value."""

    alpha: float
    optimization: OptimizeResult
    history: pd.DataFrame
    selected_candidate: PceCandidate
    selected_from_feasible_trace: bool


@dataclass
class ProgressiveBinarizationRun:
    """All PB-PCE stages and the best feasible candidate, if one exists."""

    stages: pd.DataFrame
    final_candidate: PceCandidate
    selected_candidate: PceCandidate
    selected_from_feasible_stage: bool


def minimum_pce_qubits(variable_count: int, order: int = 2) -> int:
    """Return the smallest register supporting ``variable_count`` Pauli strings."""

    if variable_count < 1:
        raise ValueError("variable_count must be positive")
    if order < 1:
        raise ValueError("order must be positive")

    qubits = order
    while 3 * comb(qubits, order) < variable_count:
        qubits += 1
    return qubits


def homogeneous_pauli_labels(num_qubits: int, order: int = 2) -> list[str]:
    """Create deterministic homogeneous X, Y, and Z Pauli strings."""

    if num_qubits < order:
        raise ValueError("num_qubits must be at least order")
    labels: list[str] = []
    for support in combinations(range(num_qubits), order):
        for axis in "XYZ":
            characters = ["I"] * num_qubits
            for qubit in support:
                # Qiskit displays the highest qubit index on the left.
                characters[num_qubits - 1 - qubit] = axis
            labels.append("".join(characters))
    return labels


def pauli_labels_for_qubo(qubo: SchedulingQubo, order: int = 2) -> tuple[int, list[str]]:
    """Allocate one unique homogeneous Pauli label to every QUBO variable."""

    qubits = minimum_pce_qubits(qubo.num_variables, order)
    labels = homogeneous_pauli_labels(qubits, order)[: qubo.num_variables]
    if len(labels) != qubo.num_variables or len(set(labels)) != len(labels):
        raise RuntimeError("could not allocate unique Pauli labels to all QUBO variables")
    return qubits, labels


def qubo_to_ising(qubo: SchedulingQubo) -> IsingModel:
    """Convert a binary QUBO exactly using the convention ``x = (s + 1) / 2``."""

    fields = np.zeros(qubo.num_variables, dtype=float)
    constant = float(qubo.offset)
    for index, coefficient in qubo.linear.items():
        fields[index] += float(coefficient) / 2.0
        constant += float(coefficient) / 2.0

    edges = sorted(qubo.quadratic)
    left = np.array([edge[0] for edge in edges], dtype=int)
    right = np.array([edge[1] for edge in edges], dtype=int)
    couplings = np.array([qubo.quadratic[edge] / 4.0 for edge in edges], dtype=float)
    for first, second, coupling in zip(left, right, couplings, strict=True):
        fields[first] += coupling
        fields[second] += coupling
        constant += coupling
    return IsingModel(constant, fields, left, right, couplings)


def ising_energy(spins: np.ndarray, model: IsingModel) -> float | np.ndarray:
    """Evaluate an Ising polynomial for one spin vector or a batch."""

    values = np.asarray(spins, dtype=float)
    return (
        model.constant
        + values @ model.fields
        + np.sum(model.couplings * values[..., model.left] * values[..., model.right], axis=-1)
    )


def decode_soft_spins(soft_spins: np.ndarray) -> np.ndarray:
    """Threshold relaxed PCE spins into QUBO bits, with zero mapped to one."""

    spins = np.where(np.asarray(soft_spins, dtype=float) >= 0.0, 1, -1)
    return ((spins + 1) // 2).astype(int)


def pce_loss_from_correlations(
    correlations: np.ndarray, alpha: float, model: IsingModel
) -> tuple[np.ndarray, float]:
    """Apply PCE binarization and evaluate the relaxed Ising loss."""

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    soft_spins = np.tanh(float(alpha) * np.asarray(correlations, dtype=float))
    return soft_spins, float(ising_energy(soft_spins, model))


def evaluate_candidate(
    theta: np.ndarray,
    alpha: float,
    correlation_evaluator: CorrelationEvaluator,
    qubo: SchedulingQubo,
    jobs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    hours: Sequence[int],
    ising_model: IsingModel | None = None,
) -> PceCandidate:
    """Evaluate, decode, and validate a PCE parameter vector."""

    model = ising_model or qubo_to_ising(qubo)
    parameter_vector = np.asarray(theta, dtype=float)
    correlations = np.asarray(correlation_evaluator(parameter_vector), dtype=float)
    if correlations.shape != (qubo.num_variables,):
        raise ValueError(
            f"expected {qubo.num_variables} PCE correlations, received {correlations.shape}"
        )
    soft_spins, pce_loss = pce_loss_from_correlations(correlations, alpha, model)
    bits = decode_soft_spins(soft_spins)
    schedule = decode_qubo_sample(qubo, bits)
    validation = validate_decoded_schedule(schedule, jobs_df, clusters_df, list(hours))
    return PceCandidate(
        theta=parameter_vector.copy(),
        alpha=float(alpha),
        correlations=correlations,
        soft_spins=soft_spins,
        bits=bits,
        pce_loss=pce_loss,
        qubo_energy=qubo_energy(qubo, bits),
        schedule=schedule,
        validation=validation,
    )


def is_feasible(candidate: PceCandidate) -> bool:
    """Return whether the decoded PCE schedule passes the shared validator."""

    return bool(candidate.validation["feasible"])


def ising_loss_scale(model: IsingModel) -> float:
    """Return a positive normalization scale for numerical optimizers."""

    return float(max(1.0, abs(model.constant) + np.abs(model.fields).sum() + np.abs(model.couplings).sum()))


def run_fixed_alpha(
    alpha: float,
    initial_theta: np.ndarray,
    correlation_evaluator: CorrelationEvaluator,
    qubo: SchedulingQubo,
    jobs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    hours: Sequence[int],
    config: FixedAlphaConfig = FixedAlphaConfig(),
    ising_model: IsingModel | None = None,
) -> FixedAlphaRun:
    """Optimize one fixed-alpha PCE problem and retain the best feasible sample."""

    model = ising_model or qubo_to_ising(qubo)
    scale = ising_loss_scale(model)
    history: list[dict[str, Any]] = []
    best_feasible: PceCandidate | None = None

    def objective(theta: np.ndarray) -> float:
        nonlocal best_feasible
        candidate = evaluate_candidate(
            theta, alpha, correlation_evaluator, qubo, jobs_df, clusters_df, hours, model
        )
        history.append(
            {
                "evaluation": len(history) + 1,
                "pce_loss": candidate.pce_loss,
                "qubo_energy": candidate.qubo_energy,
                "physical_feasible": is_feasible(candidate),
                "violation_count": len(candidate.validation["violations"]),
            }
        )
        if is_feasible(candidate) and (
            best_feasible is None or candidate.qubo_energy < best_feasible.qubo_energy
        ):
            best_feasible = candidate
        return candidate.pce_loss / scale

    optimizer_name = config.optimizer.upper()
    if optimizer_name == "COBYLA":
        optimizer_options = {
            "maxiter": config.maxiter,
            "rhobeg": config.rhobeg,
            "tol": config.tolerance,
        }
    elif optimizer_name == "POWELL":
        optimizer_options = {
            "maxiter": config.maxiter,
            "maxfev": config.maxiter,
            "xtol": config.tolerance,
            "ftol": config.tolerance,
        }
    else:
        optimizer_options = {
            "maxiter": config.maxiter,
            "rhobeg": config.rhobeg,
            "tol": config.tolerance,
        }

    optimization = minimize(
        objective,
        np.asarray(initial_theta, dtype=float),
        method=optimizer_name,
        options=optimizer_options,
    )
    final_candidate = evaluate_candidate(
        optimization.x, alpha, correlation_evaluator, qubo, jobs_df, clusters_df, hours, model
    )
    if is_feasible(final_candidate) and (
        best_feasible is None or final_candidate.qubo_energy < best_feasible.qubo_energy
    ):
        best_feasible = final_candidate
    return FixedAlphaRun(
        alpha=float(alpha),
        optimization=optimization,
        history=pd.DataFrame(history),
        selected_candidate=best_feasible or final_candidate,
        selected_from_feasible_trace=best_feasible is not None,
    )


def run_fixed_alpha_grid(
    alphas: Sequence[float],
    initial_theta: np.ndarray,
    correlation_evaluator: CorrelationEvaluator,
    qubo: SchedulingQubo,
    jobs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    hours: Sequence[int],
    config: FixedAlphaConfig = FixedAlphaConfig(),
    ising_model: IsingModel | None = None,
) -> tuple[list[FixedAlphaRun], FixedAlphaRun]:
    """Run an alpha calibration grid and select its best feasible result."""

    if not alphas:
        raise ValueError("alphas must not be empty")
    model = ising_model or qubo_to_ising(qubo)
    runs = [
        run_fixed_alpha(
            alpha, initial_theta, correlation_evaluator, qubo, jobs_df, clusters_df,
            hours, config, model,
        )
        for alpha in alphas
    ]
    feasible_runs = [run for run in runs if is_feasible(run.selected_candidate)]
    selected = min(
        feasible_runs or runs,
        key=lambda run: (
            run.selected_candidate.qubo_energy
            if feasible_runs else run.selected_candidate.pce_loss
        ),
    )
    return runs, selected


def run_progressive_binarization(
    initial_theta: np.ndarray,
    correlation_evaluator: CorrelationEvaluator,
    qubo: SchedulingQubo,
    jobs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    hours: Sequence[int],
    config: ProgressiveBinarizationConfig = ProgressiveBinarizationConfig(),
    ising_model: IsingModel | None = None,
) -> ProgressiveBinarizationRun:
    """Run PB-PCE with warm starts and the paper's adaptive alpha update rule."""

    if not 0 < config.binarization_target < 1:
        raise ValueError("binarization_target must be in (0, 1)")
    model = ising_model or qubo_to_ising(qubo)
    alpha = float(config.initial_alpha)
    theta = np.asarray(initial_theta, dtype=float).copy()
    stages: list[dict[str, Any]] = []
    best_feasible: PceCandidate | None = None
    final_candidate: PceCandidate | None = None

    for stage in range(1, config.max_stages + 1):
        fixed_run = run_fixed_alpha(
            alpha, theta, correlation_evaluator, qubo, jobs_df, clusters_df,
            hours, config.fixed_alpha, model,
        )
        candidate = fixed_run.selected_candidate
        final_candidate = candidate
        if is_feasible(candidate) and (
            best_feasible is None or candidate.qubo_energy < best_feasible.qubo_energy
        ):
            best_feasible = candidate

        magnitudes = np.abs(candidate.soft_spins)
        unbinarized = np.flatnonzero(magnitudes < config.binarization_target)
        stages.append(
            {
                "stage": stage,
                "alpha": alpha,
                "optimizer_success": bool(fixed_run.optimization.success),
                "optimizer_evaluations": int(fixed_run.optimization.nfev),
                "pce_loss": candidate.pce_loss,
                "decoded_qubo_energy": candidate.qubo_energy,
                "physical_feasible": is_feasible(candidate),
                "violation_count": len(candidate.validation["violations"]),
                "binarized_fraction": float(np.mean(magnitudes >= config.binarization_target)),
                "min_soft_spin_magnitude": float(magnitudes.min()),
            }
        )
        theta = fixed_run.optimization.x
        if not len(unbinarized):
            break

        # Algorithm 1: take the unbinarized value nearest the target M.
        target_index = unbinarized[np.argmax(magnitudes[unbinarized])]
        magnitude = float(np.clip(magnitudes[target_index], 1e-12, 1.0 - 1e-12))
        factor = np.arctanh(config.binarization_target) / np.arctanh(magnitude)
        alpha = min(config.alpha_max, max(alpha * factor, alpha * config.minimum_alpha_growth))

    if final_candidate is None:  # Defensive guard for invalid stage settings.
        raise RuntimeError("PB-PCE did not execute any stages")
    return ProgressiveBinarizationRun(
        stages=pd.DataFrame(stages),
        final_candidate=final_candidate,
        selected_candidate=best_feasible or final_candidate,
        selected_from_feasible_stage=best_feasible is not None,
    )


def _qiskit_components() -> tuple[Any, Any, Any, Any]:
    """Import optional Qiskit components only when a circuit is requested."""

    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import Pauli, Statevector
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        raise ImportError("Install the 'quantum' extra to build PCE circuits") from error
    return QuantumCircuit, ParameterVector, Pauli, Statevector


def _validate_ansatz_dimensions(num_qubits: int, layers: int) -> None:
    """Validate common PCE ansatz dimensions."""

    if num_qubits < 2:
        raise ValueError("PCE ansatz requires at least two qubits")
    if layers < 1:
        raise ValueError("layers must be positive")


def build_legacy_chain_ansatz(num_qubits: int, layers: int = 1) -> tuple[Any, list[Any]]:
    """Build the original project Ry/Rz plus sequential-CNOT-chain ansatz."""

    _validate_ansatz_dimensions(num_qubits, layers)
    QuantumCircuit, ParameterVector, _, _ = _qiskit_components()
    parameters = ParameterVector("legacy_theta", 2 * num_qubits * (layers + 1))
    circuit = QuantumCircuit(num_qubits)
    cursor = 0
    for layer in range(layers + 1):
        for qubit in range(num_qubits):
            circuit.ry(parameters[cursor], qubit)
            circuit.rz(parameters[cursor + 1], qubit)
            cursor += 2
        if layer < layers:
            for control in range(num_qubits - 1):
                circuit.cx(control, control + 1)
    return circuit, list(parameters)


def build_ibm_brickwork_ansatz(num_qubits: int, layers: int = 1) -> tuple[Any, list[Any]]:
    """Build an open staggered-CNOT brickwork ansatz for IBM experiments."""

    _validate_ansatz_dimensions(num_qubits, layers)
    QuantumCircuit, ParameterVector, _, _ = _qiskit_components()
    parameters = ParameterVector("ibm_theta", 2 * num_qubits * layers)
    circuit = QuantumCircuit(num_qubits)
    cursor = 0
    for _ in range(layers):
        for qubit in range(num_qubits):
            circuit.ry(parameters[cursor], qubit)
            circuit.rz(parameters[cursor + 1], qubit)
            cursor += 2
        for control in range(0, num_qubits - 1, 2):
            circuit.cx(control, control + 1)
        for control in range(1, num_qubits - 1, 2):
            circuit.cx(control, control + 1)
    return circuit, list(parameters)


def build_taylor_efficient_ansatz(num_qubits: int, layers: int = 1) -> tuple[Any, list[Any]]:
    """Build a Taylor-efficient-style rotating-axis trainable-entangler ansatz."""

    _validate_ansatz_dimensions(num_qubits, layers)
    QuantumCircuit, ParameterVector, _, _ = _qiskit_components()
    rotation_methods = ("rz", "rx", "ry")
    entangler_methods = ("rxx", "ryy", "rzz")
    parameter_count = sum(
        num_qubits + len(range(layer % 2, num_qubits - 1, 2))
        for layer in range(layers)
    )
    parameters = ParameterVector("taylor_theta", parameter_count)
    circuit = QuantumCircuit(num_qubits)
    cursor = 0
    for layer in range(layers):
        rotation = getattr(circuit, rotation_methods[layer % len(rotation_methods)])
        for qubit in range(num_qubits):
            rotation(parameters[cursor], qubit)
            cursor += 1
        entangler = getattr(circuit, entangler_methods[layer % len(entangler_methods)])
        for left in range(layer % 2, num_qubits - 1, 2):
            entangler(parameters[cursor], left, left + 1)
            cursor += 1
    return circuit, list(parameters)


ANSATZ_BUILDERS: dict[str, dict[str, Any]] = {
    "legacy_chain_ry_rz_cx_v1": {
        "builder": build_legacy_chain_ansatz,
        "hardware_ready": True,
        "description": "Original project Ry/Rz plus sequential CNOT chain.",
    },
    "ibm_brickwork_ry_rz_cx_open_v1": {
        "builder": build_ibm_brickwork_ansatz,
        "hardware_ready": True,
        "description": "Open staggered-CNOT brickwork designed for IBM transpilation.",
    },
    "taylor_efficient_statevector_v1": {
        "builder": build_taylor_efficient_ansatz,
        "hardware_ready": False,
        "description": "Taylor-efficient-style rotating Pauli axes and RXX/RYY/RZZ gates.",
    },
}


def build_statevector_correlation_evaluator(
    circuit: Any, parameters: Sequence[Any], labels: Sequence[str]
) -> CorrelationEvaluator:
    """Return an exact statevector correlation evaluator for a PCE circuit."""

    _, _, Pauli, Statevector = _qiskit_components()
    parameter_list = list(parameters)
    paulis = [Pauli(label) for label in labels]

    def evaluate(theta: np.ndarray) -> np.ndarray:
        values = np.asarray(theta, dtype=float)
        if values.shape != (len(parameter_list),):
            raise ValueError(f"expected {len(parameter_list)} parameters, received {values.shape}")
        bound = circuit.assign_parameters(dict(zip(parameter_list, values, strict=True)))
        state = Statevector.from_instruction(bound)
        return np.array([state.expectation_value(pauli).real for pauli in paulis])

    return evaluate
