#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define EPSILON 1e-10

// Structure to hold similarity scores and indices
typedef struct
{
    float similarity;
    int index;
} SimilarityPair;

// Comparison function for qsort
int compare_pairs(const void *a, const void *b)
{
    SimilarityPair *pair_a = (SimilarityPair *)a;
    SimilarityPair *pair_b = (SimilarityPair *)b;
    // Sort in descending order
    if (pair_a->similarity > pair_b->similarity)
        return -1;
    if (pair_a->similarity < pair_b->similarity)
        return 1;
    return 0;
}

/**
 * Calculate the cosine similarity between two vectors.
 *
 * Compile command: gcc -shared -fPIC -o cosine_lib.o cosine_similarity.c
 *
 * @param arr1: First vector.
 * @param arr2: Second vector.
 * @return Cosine similarity between the two vectors.
 */
float cosine_similarity(const float arr1[], const float arr2[], int vector_dim)
{
    float dot_product = 0.0f, norm1 = 0.0f, norm2 = 0.0f;

    // Calculate dot product and norms in a single pass
    for (int i = 0; i < vector_dim; i++)
    {
        dot_product += arr1[i] * arr2[i];
        norm1 += arr1[i] * arr1[i];
        norm2 += arr2[i] * arr2[i];
    }

    // Calculate magnitudes
    norm1 = sqrt(norm1);
    norm2 = sqrt(norm2);

    // Avoid division by zero
    if (norm1 < EPSILON || norm2 < EPSILON)
    {
        return 0.0f;
    }

    // Calculate cosine similarity
    return dot_product / (norm1 * norm2);
}

// Function to find top K similar vectors
void find_top_similar(const float *query,
                      const float *database,
                      int vector_dim,
                      int num_vectors,
                      int *top_indices,
                      int num_threads,
                      int k)
{
    if (num_threads > 0)
    {
        omp_set_num_threads(num_threads);
    }

    // Allocate array for similarities using num_vectors
    SimilarityPair *similarities = (SimilarityPair *)malloc(num_vectors * sizeof(SimilarityPair));
    if (!similarities)
    {
        fprintf(stderr, "Memory allocation failed\n");
        return;
    }

// Use num_vectors in parallel loop
#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < num_vectors; i++)
    {
        similarities[i].similarity = cosine_similarity(query,
                                                       database + (i * vector_dim),
                                                       vector_dim);
        similarities[i].index = i;
    }

    // Sort using num_vectors
    qsort(similarities, num_vectors, sizeof(SimilarityPair), compare_pairs);

    // Copy top K indices, ensuring we don't exceed num_vectors
    k = (k < num_vectors) ? k : num_vectors;
    for (int i = 0; i < k; i++)
    {
        top_indices[i] = similarities[i].index;
    }

    free(similarities);
}

/**
 * Find top K similar vectors for multiple queries in batch
 *
 * @param queries: Array of query vectors [num_queries × vector_dim]
 * @param database: Array of database vectors [num_vectors × vector_dim]
 * @param vector_dim: Dimension of each vector
 * @param num_vectors: Number of vectors in database
 * @param num_queries: Number of query vectors
 * @param top_indices: Output array for top K indices [num_queries × TOP_K]
 * @param num_threads: Number of threads to use (0 for default)
 */
void find_top_similar_batch(const float *queries,
                            const float *database,
                            int vector_dim,
                            int num_vectors,
                            int num_queries,
                            int *top_indices,
                            int num_threads,
                            int k)
{
    if (num_threads > 0)
    {
        omp_set_num_threads(num_threads);
    }

// Process each query in parallel
#pragma omp parallel for schedule(dynamic, 1)
    for (int q = 0; q < num_queries; q++)
    {
        // Allocate similarities array for this query
        SimilarityPair *similarities = (SimilarityPair *)malloc(num_vectors * sizeof(SimilarityPair));
        if (!similarities)
        {
            fprintf(stderr, "Memory allocation failed for query %d\n", q);
            continue;
        }

        // Get pointer to current query vector
        const float *query = queries + (q * vector_dim);

// Calculate similarities for all database vectors
#pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < num_vectors; i++)
        {
            similarities[i].similarity = cosine_similarity(query,
                                                           database + (i * vector_dim),
                                                           vector_dim);
            similarities[i].index = i;
        }

        // Sort similarities for this query
        qsort(similarities, num_vectors, sizeof(SimilarityPair), compare_pairs);

        // Copy top K indices for this query
        k = (k < num_vectors) ? k : num_vectors;
        for (int i = 0; i < k; i++)
        {
            // Store in the appropriate position in the output array
            top_indices[q * k + i] = similarities[i].index;
        }

        free(similarities);
    }
}