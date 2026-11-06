"""Persist the four locked sweep states and invalidate outdated context."""
from library_claim.domain.models import Sweep
from library_claim.storage.repository import Repository

NEXT_STATE = {
    'Sweeping': 'Approaching',
    'Approaching': 'Capturing',
    'Capturing': 'Resuming',
    'Resuming': 'Sweeping',
}


class SweepController:
    """Apply explicit transitions through one repository owned by the caller."""

    def __init__(self, repository: Repository, sweep_id: str):
        """Bind the controller to a persisted sweep.

        Args:
            repository: Repository used on its owning thread.
            sweep_id: Identifier of the sweep to control.
        """
        self.repository = repository
        self.sweep_id = sweep_id
        repository.get(Sweep, sweep_id)

    def transition(self, state: str, *, target_id: str | None = None) -> Sweep:
        """Advance one step and invalidate results based on previous context.

        Args:
            state: The next state in the locked cycle.
            target_id: Region or book identifier, required when approaching.

        Returns:
            The updated sweep.

        Raises:
            ValueError: The transition or target is invalid.
        """
        sweep = self.repository.get(Sweep, self.sweep_id)
        if NEXT_STATE[sweep.state] != state:
            raise ValueError(f'Cannot transition from {sweep.state} to {state}')
        if state == 'Approaching' and not target_id:
            raise ValueError('Approaching requires a target')
        if state != 'Approaching' and target_id is not None:
            raise ValueError('The target can only be selected when approaching')
        target = target_id if state == 'Approaching' else sweep.active_target
        if state == 'Sweeping':
            target = None
        return self.repository.update(sweep.model_copy(update={
            'state': state, 'active_target': target,
            'context_revision': sweep.context_revision + 1,
        }))

    def invalidate_context(self) -> Sweep:
        """Invalidate in-flight reasoning after a correction or changed view.

        Returns:
            The sweep with an incremented context revision.
        """
        sweep = self.repository.get(Sweep, self.sweep_id)
        return self.repository.update(sweep.model_copy(update={
            'context_revision': sweep.context_revision + 1,
        }))

    def recover(self) -> Sweep:
        """Reset interrupted capture to Sweeping while preserving inventory.

        Returns:
            The sweep ready to reacquire visual context.
        """
        sweep = self.repository.get(Sweep, self.sweep_id)
        return self.repository.update(sweep.model_copy(update={
            'state': 'Sweeping', 'active_target': None,
            'context_revision': sweep.context_revision + 1,
        }))
