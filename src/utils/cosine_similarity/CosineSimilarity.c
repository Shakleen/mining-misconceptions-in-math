#include <stdio.h>
#include <math.h>

#define EPSILON 1e-10

/**
 * Calculate the cosine similarity between two vectors.
 *
 * @param arr1: First vector.
 * @param arr2: Second vector.
 * @param size: Size of the input arrays.
 * @return Cosine similarity between the two vectors.
 */
float cosine_similarity(const float arr1[], const float arr2[], const int size)
{
    float dot_product = 0.0f, norm1 = 0.0f, norm2 = 0.0f;

    // Calculate dot product and norms in a single pass
    for (int i = 0; i < size; i++)
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