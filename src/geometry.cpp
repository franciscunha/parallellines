#include "geometry.hpp"

Matrix4::Matrix4() {
    m = {
        std::array<float, 4> {1.0f, 0.0f, 0.0f, 0.0f},
        std::array<float, 4> {0.0f, 1.0f, 0.0f, 0.0f},
        std::array<float, 4> {0.0f, 0.0f, 1.0f, 0.0f},
        std::array<float, 4> {0.0f, 0.0f, 0.0f, 1.0f}
    };
}

Matrix4 Matrix4::identity() {
    return Matrix4();
}

Matrix4 Matrix4::projection(float c) {
    Matrix4 _m = Matrix4();
    _m.m[3][2] = -(1/c);
    return _m;
}

Matrix4 Matrix4::operator *(const Matrix4& rhs) {
    Matrix4 result = Matrix4();
    const Matrix4& lhs = *this; // alias for clarity
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {

            result.m[i][j] = 0;

            for (int k = 0; k < 4; k++) {
                result.m[i][j] += lhs.m[i][k] * rhs.m[k][j];
            }
            
        }
    }

    return result;
}

Vec4f   Matrix4::operator *(const Vec4f& v) {
    Vec4f result = Vec4f(0, 0, 0, 0);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result.raw[i] += v.raw[j] * m[i][j];
        }
    }
    return result;
}

std::ostream& operator<<(std::ostream& s, Matrix4& m) {
	s << "|" << m.m[0][0] << ", " << m.m[0][1] << ", " << m.m[0][2] << ", " << m.m[0][3] << "|" << std::endl;
	s << "|" << m.m[1][0] << ", " << m.m[1][1] << ", " << m.m[1][2] << ", " << m.m[1][3] << "|" << std::endl;
	s << "|" << m.m[2][0] << ", " << m.m[2][1] << ", " << m.m[2][2] << ", " << m.m[2][3] << "|" << std::endl;
	s << "|" << m.m[3][0] << ", " << m.m[3][1] << ", " << m.m[3][2] << ", " << m.m[3][3] << "|" << std::endl;
	return s;
}

Matrix3::Matrix3() {
    m = {
        std::array<float, 3> {1.0f, 0.0f, 0.0f},
        std::array<float, 3> {0.0f, 1.0f, 0.0f},
        std::array<float, 3> {0.0f, 0.0f, 1.0f}
    };
}

Matrix3 Matrix3::identity() {
    return Matrix3();
}

Matrix4 Matrix3::homogenize() {
    Matrix4 m4 = Matrix4();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            m4.m[i][j] = m[i][j];
        }
    }
    return m4;
}

Matrix3 Matrix3::operator *(const Matrix3& rhs) {
    Matrix3 result = Matrix3();
    const Matrix3& lhs = *this; // alias for clarity
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {

            result.m[i][j] = 0;

            for (int k = 0; k < 3; k++) {
                result.m[i][j] += lhs.m[i][k] * rhs.m[k][j];
            }
            
        }
    }

    return result;
}

Vec3f   Matrix3::operator *(const Vec3f& v) {
    Vec3f result = Vec3f(0, 0, 0);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result.raw[i] += v.raw[j] * m[i][j];
        }
    }
    return result;
}


std::ostream& operator<<(std::ostream& s, Matrix3& m) {
	s << "|" << m.m[0][0] << ", " << m.m[0][1] << ", " << m.m[0][2] << "|" << std::endl;
	s << "|" << m.m[1][0] << ", " << m.m[1][1] << ", " << m.m[1][2] << "|" << std::endl;
	s << "|" << m.m[2][0] << ", " << m.m[2][1] << ", " << m.m[2][2] << "|" << std::endl;
	return s;
}