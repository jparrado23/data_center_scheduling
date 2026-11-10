"""IBM Runtime adapters for PCE correlation evaluation.

This module contains transport and transpilation concerns only.  PB-PCE
continuation and schedule validation remain in :mod:`src.quantum.pce`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass
class IbmPceCorrelationEvaluator:
    """Callable IBM estimator wrapper that records the most recent job ID."""

    estimator: Any
    isa_ansatz: Any
    isa_observables: list[Any]
    expected_correlations: int
    last_job_id: str | None = None

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        """Submit one parameter point and return its Pauli expectations."""

        job = self.estimator.run(
            [(self.isa_ansatz, self.isa_observables, [np.asarray(theta, dtype=float)])]
        )
        result = job.result()[0]
        correlations = np.asarray(result.data.evs, dtype=float).reshape(-1)
        if correlations.shape != (self.expected_correlations,):
            raise RuntimeError(
                f"expected {self.expected_correlations} correlations, received {correlations.shape}"
            )
        self.last_job_id = job.job_id()
        return correlations


def create_ibm_service(token: str, instance: str | None = None) -> Any:
    """Create an IBM Quantum Runtime service without persisting credentials."""

    if not token:
        raise ValueError("an IBM Quantum API token is required")
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        raise ImportError("Install the 'ibm-qpu' extra for IBM PCE execution") from error
    options: dict[str, str] = {"channel": "ibm_quantum_platform", "token": token}
    if instance:
        options["instance"] = instance
    return QiskitRuntimeService(**options)


def select_qpu_backend(service: Any, minimum_qubits: int, backend_name: str | None = None) -> Any:
    """Select a requested or least-busy operational IBM QPU of sufficient size."""

    backend = (
        service.backend(backend_name)
        if backend_name
        else service.least_busy(operational=True, simulator=False, min_num_qubits=minimum_qubits)
    )
    if backend.num_qubits < minimum_qubits:
        raise ValueError(
            f"backend {backend.name} has {backend.num_qubits} qubits; "
            f"at least {minimum_qubits} are required"
        )
    return backend


def build_ibm_correlation_evaluator(
    backend: Any,
    circuit: Any,
    labels: Sequence[str],
    *,
    precision: float = 0.03,
    optimization_level: int = 1,
) -> IbmPceCorrelationEvaluator:
    """Transpile one PCE circuit and return an IBM estimator-backed evaluator."""

    if precision <= 0:
        raise ValueError("precision must be positive")
    try:
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime.executor_estimator import Estimator
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        raise ImportError("Install the 'ibm-qpu' extra for IBM PCE execution") from error

    pass_manager = generate_preset_pass_manager(
        backend=backend, optimization_level=optimization_level
    )
    isa_ansatz = pass_manager.run(circuit)
    observables = [
        SparsePauliOp(label).apply_layout(isa_ansatz.layout)
        for label in labels
    ]
    estimator = Estimator(mode=backend)
    estimator.options.default_precision = precision
    return IbmPceCorrelationEvaluator(
        estimator=estimator,
        isa_ansatz=isa_ansatz,
        isa_observables=observables,
        expected_correlations=len(labels),
    )
