# cython: initializedcheck=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

cdef int bfprt(int[:] items, float[:] scores, int left, int right, int k):
    cdef int index
    if left < right:
        index = partition(items, scores, left, right)
        
        if k == index:
            return index
        elif k < index:
            return bfprt(items, scores, left, index - 1, k)
        else:
            return bfprt(items, scores, index + 1, right, k)
        
    return left

cdef int partition(int[:] items, float[:] scores, int left, int right):
    cdef float datum = scores[findMid(items, scores, left, right)]
    cdef int i = left
    cdef int j = right
    
    while i < j:
        while scores[i] < datum:
            i += 1
        while scores[j] > datum:
            j -= 1
            
        if i < j:
            swap(items, scores, i, j)
            
        while scores[i] == scores[j] and i != j:
            # i += 1
            j -= 1
    return i

cdef int findMid(int[:] items, float[:] scores, int left, int right):
    if right - left < 5:
        return insertSort(items, scores, left, right)
    
    cdef int n = left - 1
    
    cdef int i = left
    cdef int index
    while i + 4 <= right:
        index = insertSort(items, scores, i, i + 4)
        n += 1
        swap(items, scores, n, index)
        i += 5
    return findMid(items, scores, left, n)

cdef int insertSort(int[:] items, float[:] scores, int left, int right):
    cdef int temp1
    cdef float temp2
    cdef int j
    cdef int i = left + 1
    while i <= right:
        temp2 = scores[i]
        temp1 = items[i]
        j = i - 1
        while j >= left and scores[j] > temp2:
            scores[j + 1] = scores[j]
            items[j + 1] = items[j]
            j -= 1
        scores[j + 1] = temp2
        items[j + 1] = temp1
        i += 1
        
    return ((right - left) >> 1) + left
        
        
cdef inline swap(int[:] items, float[:] scores, int i, int j):
    items[i], items[j] = items[j], items[i]
    scores[i], scores[j] = scores[j], scores[i]

cpdef ksmallest(int k, int[:] items, float[:] scores):
    bfprt(items, scores, 0, len(items) - 1, k)
    # return scores[: bfprt(items, scores, 0, len(items) - 1, k)]
    # return bfprt(items, scores, 0, len(items) - 1, k)
    return items[:k], scores[:k]
