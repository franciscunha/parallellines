#include "../include/geometry.cuh"

__host__ __device__ Matrix4::Matrix4()
{
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            raw[i * 4 + j] = i == j ? 1.0f : 0.0f;
        }
    }
}

__host__ __device__ Matrix4 Matrix4::identity()
{
    return Matrix4();
}

__host__ __device__ Matrix4 Matrix4::transpose()
{
    Matrix4 t;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            t.raw[i * 4 + j] = raw[j * 4 + i];
        }
    }
    return t;
}

__host__ __device__ float Matrix4::cofactor(int i_, int j_)
{
    float submatrix[9];

    int sub_i = 0;
    for (int i = 0; i < 4; i++)
    {
        if (i == i_)
            continue;

        int sub_j = 0;
        for (int j = 0; j < 4; j++)
        {
            if (j == j_)
                continue;

            submatrix[sub_i * 3 + sub_j] = raw[i * 4 + j];
            sub_j++;
        }
        sub_i++;
    }

    float det = Matrix3::determinant(submatrix);
    int neg = ((i_ + j_) % 2 == 0) ? 1 : -1;

    return neg * det;
}

__host__ __device__ bool Matrix4::inverse(Matrix4 &inverse)
{
    // pre-compute cofactors
    float cofactors[4][4];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            cofactors[i][j] = cofactor(i, j);
        }
    }

    // compute determinant
    float det = 0.0f;
    for (int j = 0; j < 4; j++)
    {
        det += raw[/* 0 * 4 + */ j] * cofactors[0][j];
    }
    if (std::abs(det) < 1e-6)
    {
        // det == 0 -> non-invertible matrix
        return false;
    }
    float inv_det = (1.0f / det);

    // compute adjoint
    float adj[4][4] = {
        {cofactors[0][0], cofactors[1][0], cofactors[2][0], cofactors[3][0]},
        {cofactors[0][1], cofactors[1][1], cofactors[2][1], cofactors[3][1]},
        {cofactors[0][2], cofactors[1][2], cofactors[2][2], cofactors[3][2]},
        {cofactors[0][3], cofactors[1][3], cofactors[2][3], cofactors[3][3]},
    };

    // put them together to make the inverse
    inverse = Matrix4();
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            inverse.raw[i * 4 + j] = inv_det * adj[i][j];
        }
    }

    return true;
}

__host__ __device__ Vec3f Matrix4::mult(Vec3f &v, bool is_point)
{
    return (*this * v.homogenize(is_point)).dehomogenize();
}

__host__ __device__ Matrix4 Matrix4::operator*(const Matrix4 &rhs)
{
    Matrix4 result = Matrix4();
    const Matrix4 &lhs = *this; // alias for clarity

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {

            result.raw[i * 4 + j] = 0;

            for (int k = 0; k < 4; k++)
            {
                result.raw[i * 4 + j] += lhs.raw[i * 4 + k] * rhs.raw[k * 4 + j];
            }
        }
    }

    return result;
}

__host__ __device__ Vec4f Matrix4::operator*(const Vec4f &v)
{
    Vec4f result = Vec4f(0, 0, 0, 0);
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            result.raw[i] += v.raw[j] * raw[i * 4 + j];
        }
    }
    return result;
}

std::ostream &operator<<(std::ostream &s, Matrix4 &m)
{
    s << "|" << m.get(0, 0) << ", " << m.get(0, 1) << ", " << m.get(0, 2) << ", " << m.get(0, 3) << "|" << std::endl;
    s << "|" << m.get(1, 0) << ", " << m.get(1, 1) << ", " << m.get(1, 2) << ", " << m.get(1, 3) << "|" << std::endl;
    s << "|" << m.get(2, 0) << ", " << m.get(2, 1) << ", " << m.get(2, 2) << ", " << m.get(2, 3) << "|" << std::endl;
    s << "|" << m.get(3, 0) << ", " << m.get(3, 1) << ", " << m.get(3, 2) << ", " << m.get(3, 3) << "|" << std::endl;
    return s;
}

__host__ __device__ Matrix3::Matrix3()
{
    for (int i = 0; i < 9; i++)
    {
        for (int j = 0; j < 9; j++)
        {
            raw[i * 3 + j] = i == j ? 1.0f : 0.0f;
        }
    }
}

__host__ __device__ Matrix3 Matrix3::identity()
{
    return Matrix3();
}

__host__ __device__ float Matrix3::determinant(float *m)
{
    return m[0 * 3 + 0] * (m[1 * 3 + 1] * m[2 * 3 + 2] - m[1 * 3 + 2] * m[2 * 3 + 1]) - m[0 * 3 + 1] * (m[1 * 3 + 0] * m[2 * 3 + 2] - m[1 * 3 + 2] * m[2 * 3 + 0]) + m[0 * 3 + 2] * (m[1 * 3 + 0] * m[2 * 3 + 1] - m[1 * 3 + 1] * m[2 * 3 + 0]);
}

__host__ __device__ float Matrix3::determinant()
{
    return Matrix3::determinant(raw);
}

__host__ __device__ Matrix4 Matrix3::homogenize()
{
    Matrix4 m4 = Matrix4();
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            m4.raw[i * 4 + j] = raw[i * 3 + j];
        }
    }
    return m4;
}

__host__ __device__ Matrix3 Matrix3::operator*(const Matrix3 &rhs)
{
    Matrix3 result = Matrix3();
    const Matrix3 &lhs = *this; // alias for clarity

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {

            result.raw[i * 3 + j] = 0;

            for (int k = 0; k < 3; k++)
            {
                result.raw[i * 3 + j] += lhs.raw[i * 3 + k] * rhs.raw[k * 3 + j];
            }
        }
    }

    return result;
}

__host__ __device__ Vec3f Matrix3::operator*(const Vec3f &v)
{
    Vec3f result = Vec3f(0, 0, 0);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            result.raw[i] += v.raw[j] * raw[i * 3 + j];
        }
    }
    return result;
}

std::ostream &operator<<(std::ostream &s, Matrix3 &m)
{
    s << "|" << m.get(0, 0) << ", " << m.get(0, 1) << ", " << m.get(0, 2) << "|" << std::endl;
    s << "|" << m.get(1, 0) << ", " << m.get(1, 1) << ", " << m.get(1, 2) << "|" << std::endl;
    s << "|" << m.get(2, 0) << ", " << m.get(2, 1) << ", " << m.get(2, 2) << "|" << std::endl;
    return s;
}