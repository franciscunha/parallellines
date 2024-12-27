// This file is forked from an initial version by Dmitry V. Sokolov (https://github.com/ssloy) that was
// provided as a starting point for the tinyrenderer course (https://github.com/ssloy/tinyrenderer)

#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include <cmath>
#include <array>
#include <ostream>
#include "tgaimage.cuh"

// forward declare vector types so they can reference each other
template <class t>
struct Vec2;
template <class t>
struct Vec3;
template <class t>
struct Vec4;

template <class t>
struct Vec2
{
	union
	{
		struct
		{
			t u, v;
		};
		struct
		{
			t x, y;
		};
		t raw[2];
	};
	__host__ __device__ Vec2() : u(0), v(0) {}
	__host__ __device__ Vec2(t _u, t _v) : u(_u), v(_v) {}

	__host__ __device__ inline Vec2<t> operator+(const Vec2<t> &V) const { return Vec2<t>(u + V.u, v + V.v); }
	__host__ __device__ inline Vec2<t> operator-(const Vec2<t> &V) const { return Vec2<t>(u - V.u, v - V.v); }
	__host__ __device__ inline Vec2<t> operator*(float f) const { return Vec2<t>(u * f, v * f); }
	__host__ __device__ inline t dot(const Vec2<t> &v) const { return x * v.x + y * v.y; }

	template <class>
	friend std::ostream &operator<<(std::ostream &s, Vec2<t> &v);
};

template <class t>
struct Vec3
{
	union
	{
		struct
		{
			t x, y, z;
		};
		struct
		{
			t ivert, iuv, inorm;
		};
		t raw[3];
	};
	__host__ __device__ Vec3() : x(0), y(0), z(0) {}
	__host__ __device__ Vec3(t _x, t _y, t _z) : x(_x), y(_y), z(_z) {}

	__host__ __device__ inline Vec3<t> operator+(const Vec3<t> &v) const { return Vec3<t>(x + v.x, y + v.y, z + v.z); }
	__host__ __device__ inline Vec3<t> operator-(const Vec3<t> &v) const { return Vec3<t>(x - v.x, y - v.y, z - v.z); }
	__host__ __device__ inline Vec3<t> operator*(float f) const { return Vec3<t>(x * f, y * f, z * f); }
	__host__ __device__ inline t dot(const Vec3<t> &v) const { return x * v.x + y * v.y + z * v.z; }
	__host__ __device__ inline Vec3<t> cross(const Vec3<t> &v) const { return Vec3<t>(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x); }

	__host__ __device__ inline float norm() const { return std::sqrt(x * x + y * y + z * z); }
	__host__ __device__ Vec3<t> &normalize(t l = 1)
	{
		*this = (*this) * (l / norm());
		return *this;
	}
	__host__ __device__ Vec4<t> homogenize(bool is_point = true) { return Vec4<t>(x, y, z, is_point ? 1 : 0); }
	__host__ __device__ static Vec3<t> from_tgacolor(TGAColor c) { return Vec3<t>(c.r, c.g, c.b); }

	template <class>
	friend std::ostream &operator<<(std::ostream &s, Vec3<t> &v);
};

template <class t>
struct Vec4
{
	union
	{
		struct
		{
			t x, y, z, w;
		};
		t raw[4];
	};
	__host__ __device__ Vec4() : x(0), y(0), z(0), w(0) {}
	__host__ __device__ Vec4(t _x, t _y, t _z, t _w) : x(_x), y(_y), z(_z), w(_w) {}

	__host__ __device__ inline Vec4<t> operator+(const Vec4<t> &v) const { return Vec4<t>(x + v.x, y + v.y, z + v.z, w + v.w); }
	__host__ __device__ inline Vec4<t> operator-(const Vec4<t> &v) const { return Vec4<t>(x - v.x, y - v.y, z - v.z, w - v.w); }
	__host__ __device__ inline Vec4<t> operator*(float f) const { return Vec4<t>(x * f, y * f, z * f, w * f); }
	__host__ __device__ inline t dot(const Vec4<t> &v) const { return x * v.x + y * v.y + z * v.z + w * v.w; }

	__host__ __device__ Vec3<t> dehomogenize()
	{
		if ((w >= 0 ? w : (w * -1)) < 1e-6)
		{
			return Vec3<t>(x, y, z);
		}
		return Vec3<t>(x / w, y / w, z / w);
	}
	__host__ __device__ static Vec4<t> from_tgacolor(TGAColor c) { return Vec4<t>(c.r, c.g, c.b, c.a); }

	template <class>
	friend std::ostream &operator<<(std::ostream &s, Vec3<t> &v);
};

typedef Vec2<float> Vec2f;
typedef Vec2<int> Vec2i;
typedef Vec3<float> Vec3f;
typedef Vec3<int> Vec3i;
typedef Vec4<float> Vec4f;
typedef Vec4<int> Vec4i;

template <class t>
std::ostream &operator<<(std::ostream &s, Vec2<t> &v)
{
	s << "(" << v.x << ", " << v.y << ")\n";
	return s;
}

template <class t>
std::ostream &operator<<(std::ostream &s, Vec3<t> &v)
{
	s << "(" << v.x << ", " << v.y << ", " << v.z << ")\n";
	return s;
}

template <class t>
std::ostream &operator<<(std::ostream &s, Vec4<t> &v)
{
	s << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")\n";
	return s;
}

class Matrix4
{
private:
	__host__ __device__ float cofactor(int i, int j);

public:
	float raw[16];

	__host__ __device__ Matrix4();
	__host__ __device__ static Matrix4 identity();

	__host__ __device__ inline float get(int i, int j) { return raw[i * 4 + j]; }

	__host__ __device__ Matrix4 transpose();
	__host__ __device__ bool inverse(Matrix4 &inverse);

	__host__ __device__ Vec3f mult(Vec3f &v, bool is_point = true);
	__host__ __device__ Matrix4 operator*(const Matrix4 &a);
	__host__ __device__ Vec4f operator*(const Vec4f &v);

	friend std::ostream &operator<<(std::ostream &s, Matrix4 &m);
};

class Matrix3
{
private:
	float raw[9];

public:
	__host__ __device__ Matrix3();
	__host__ __device__ static Matrix3 identity();

	__host__ __device__ inline float get(int i, int j) { return raw[i * 3 + j]; }

	__host__ __device__ static float determinant(float *m);
	__host__ __device__ float determinant();
	__host__ __device__ Matrix4 homogenize();

	__host__ __device__ Matrix3 operator*(const Matrix3 &a);
	__host__ __device__ Vec3f operator*(const Vec3f &v);

	friend std::ostream &operator<<(std::ostream &s, Matrix3 &m);
};

#endif //__GEOMETRY_H__