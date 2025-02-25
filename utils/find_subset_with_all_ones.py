import numpy as np


def find_subset_with_all_ones(matrix, row_weight=1, min_rows=5, min_cols=600):
    """
    Finds a subset of the given binary matrix with all ones,
    represented as row and column indices.

    Args:
        matrix (numpy.ndarray): A binary matrix with M rows and N columns.
        row_weight (float): Weight factor for prioritizing rows (Default: 1.0).
        min_rows (int):  Minimum desired number of rows in the subset (Default: 10).
        min_cols (int):  Minimum desired number of columns in the subset (Default: 50).

    Returns:
        A tuple of two lists: (row_inds, col_inds), where
        row_inds is a list of row indices of the subset, and
        col_inds is a list of column indices of the subset.
    """
    M, N = matrix.shape
    if M == 0 or N == 0:
        return None, None
    row_inds = list(range(M))
    col_inds = list(range(N))
    # first remove any rows or columns that are all 0s
    row_ones = np.sum(matrix == 1, axis=1)
    col_ones = np.sum(matrix == 1, axis=0)
    row_inds = [i for i in row_inds if row_ones[i] > 0]
    col_inds = [i for i in col_inds if col_ones[i] > 0]
    # only keep rows and columns that have at least one 1 in matrix
    matrix = matrix[row_inds, :]
    matrix = matrix[:, col_inds]
    M = len(row_inds)
    N = len(col_inds)

    while True:
        # Find the rows and columns with the largest number of 0s
        if len(row_inds) == 0 or len(col_inds) == 0:
            return None, None
        row_zeros = np.sum(matrix == 0, axis=1)
        col_zeros = np.sum(matrix == 0, axis=0)
        max_row_zeros = np.max(row_zeros)
        max_col_zeros = np.max(col_zeros)

        # trials are on order of 800. neurons order of 50.
        # to weigh both equally, neurons should get a multiplier of 800/50 = 16
        proportion_row_zeros = max_row_zeros
        proportion_col_zeros = max_col_zeros * N / M * row_weight

        if max_row_zeros == 0 and max_col_zeros == 0:
            # No more 0s left in the matrix, we're done
            break

        if len(row_inds) < min_rows:
            # keep the last few rows so we have a decent number of neurons
            # to work with. this means we have to remove trials.
            col_to_remove = np.argmax(col_zeros)
            matrix = np.delete(matrix, col_to_remove, axis=1)
            col_inds.remove(col_inds[col_to_remove])
            continue

        if (
            proportion_row_zeros >= proportion_col_zeros
            or len(col_inds) < min_cols
        ):
            # Remove the row with the largest number of 0s
            row_to_remove = np.argmax(row_zeros)
            matrix = np.delete(matrix, row_to_remove, axis=0)
            row_inds.remove(row_inds[row_to_remove])
        else:
            # Remove the column with the largest number of 0s
            col_to_remove = np.argmax(col_zeros)
            matrix = np.delete(matrix, col_to_remove, axis=1)
            col_inds.remove(col_inds[col_to_remove])
    return row_inds, col_inds
