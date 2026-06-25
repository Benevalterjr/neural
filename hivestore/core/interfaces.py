import abc
import numpy as np

class IVectorStore(abc.ABC):
    @abc.abstractmethod
    def append_vector(self, vec: np.ndarray) -> int:
        """Appends a vector to the store and returns its index/offset."""
        pass

    @abc.abstractmethod
    def append_graph_edges(self, neighbors: list) -> int:
        """Appends graph edges (neighbors) and returns their offset."""
        pass

    @abc.abstractmethod
    def write_cell_meta(self, cell_id: int, v_off: int, n_off: int, n_count: int) -> None:
        """Writes metadata for a specific cell ID."""
        pass

    @abc.abstractmethod
    def read_cell_meta(self, cell_id: int) -> dict:
        """Reads metadata for a specific cell ID."""
        pass

    @abc.abstractmethod
    def read_vector(self, idx: int) -> np.ndarray:
        """Reads a vector at a specific index."""
        pass

    @abc.abstractmethod
    def read_neighbors(self, meta: dict) -> np.ndarray:
        """Reads neighbors based on metadata info."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the store."""
        pass
