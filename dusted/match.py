import numba
import numpy as np


@numba.njit()
def match(x, y, sim, gap):
    """Find a similar unit sub-sequence between two utterances.

    Args:
        x (NDArray): discrete units for the first utterance of shape (N,).
        y (NDArray): discrete units for the second utterance of shape (M,).
        sim (NDArray): substitution function that returns a score for matching units of shape (K, K) where K is the total number of discrete units.
        gap (float): the gap penalty.

    Returns:
        NDArray[Tuple(int,int)]: list of aligned indices in x and y.
        NDArray[int]: matching sub-sequence in x.
        NDArray[int]: matching sub-sequence in y.
        float: similarity score.
    """
    H, T = score(x, y, sim, gap)
    similarity = np.max(H)
    starts = np.argwhere(H == similarity)
    start = starts[starts.sum(axis=-1).argmin()]

    path, a, b = backtrace(T, start, x, y)
    return path, a, b, similarity


@numba.njit()
def rescore(H, T, x, y, sim, gap, visited, start):
    M = np.copy(H)
    scores = np.zeros(4, dtype=np.float32)

    istart, jstart = start
    jend = jstart

    for i in range(istart, M.shape[0]):
        jinc = jstart
        jmatched = False

        for j in range(jstart, M.shape[1]):
            if visited[i, j]:
                M[i, j] = 0
                T[i, j] = 0
                continue

            scores[1] = M[i - 1, j - 1] + sim[x[i - 1], y[j - 1]]
            scores[2] = M[i - 1, j] - gap
            scores[3] = M[i, j - 1] - gap
            k = np.argmax(scores)
            M[i, j] = scores[k]
            T[i, j] = k

            if M[i, j] == H[i, j]:
                if j == jinc:
                    jstart += 1
                elif j >= jend:
                    jmatched = True
                    jend = j
                    break

        if not jmatched:
            jend = M.shape[1] - 1

        if jinc == jend:
            break
    return M, T


@numba.njit()
def match_rescore(
    x: np.ndarray, y: np.ndarray, sub: np.ndarray, gap: float, threshold: float = 6
):
    """Find similar unit sub-sequences between two utterances.

    Args:
        x (NDArray): discrete units for the first utterance of shape (N,).
        y (NDArray): discrete units for the second utterance of shape (M,).
        sim (NDArray): substitution function that returns a score for matching units of shape (K, K) where K is the total number of discrete units.
        gap (float): the gap penalty.
        tau (float): similarity threshold for matches (defaults to 6).

    Yields:
        NDArray[Tuple(int,int)]: list of aligned indices in x and y.
        NDArray[int]: matching sub-sequence in x.
        NDArray[int]: matching sub-sequence in y.
        float: similarity score.

    Notes:
        The function finds multiple matches by recomputing the scoring matrix `H` after each match is found.
        This allows the discovery of secondary matches that are locally optimal but do not overlap with previously identified matches.
    """
    H, T = score(x, y, sub, gap)

    visited = np.zeros_like(H, dtype=np.bool_)

    while True:
        similarity = np.max(H)

        if similarity < threshold:
            break

        starts = np.argwhere(H == similarity)
        start = starts[starts.sum(axis=-1).argmin()]

        path, a, b = backtrace(T, start, x, y)

        yield path, a, b, similarity

        for i, j in path:
            visited[i, j] = True

        H, T = rescore(H, T, x, y, sub, gap, visited, path[0])
        similarity = np.max(H)


@numba.njit()
def score(x, y, sim, gap):
    n, m = len(x), len(y)
    H = np.zeros((n + 1, m + 1), dtype=np.float32)
    T = np.full((n + 1, m + 1), 0, dtype=np.int16)

    scores = np.zeros(4, dtype=np.float32)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            scores[1] = H[i - 1, j - 1] + sim[x[i - 1], y[j - 1]]
            scores[2] = H[i - 1, j] - gap
            scores[3] = H[i, j - 1] - gap
            k = np.argmax(scores)
            H[i, j] = scores[k]
            T[i, j] = k

    return H, T


@numba.njit()
def backtrace(T, start, x, y, blank=-1):
    i, j = start
    path = []

    a = []
    b = []

    while T[i, j] != 0 and (i > 0 or j > 0):
        path.append((i, j))

        if T[i, j] == 1:  # substitution
            i -= 1
            j -= 1
            a.append(x[i])
            b.append(y[j])
        elif T[i, j] == 2:  # deletion
            i -= 1
            a.append(x[i])
            b.append(blank)
        elif T[i, j] == 3:  # insertion
            j -= 1
            a.append(blank)
            b.append(y[j])

    path.reverse()
    a.reverse()
    b.reverse()
    return path, a, b
