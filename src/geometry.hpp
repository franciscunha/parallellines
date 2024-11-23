// This file is forked from an initial version by Dmitry V. Sokolov (https://github.com/ssloy) that was
// provided as a starting point for the tinyrenderer course (https://github.com/ssloy/tinyrenderer)

#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include <cmath>
#include <array>
#include <ostream>

// forward declare vector types so they can reference each other
template <class t> struct Vec2;
template <class t> struct Vec3;
template <class t> struct Vec4;

template <class t> struct Vec2 {
	union {
		struct {t u, v;};
		struct {t x, y;};
		t raw[2];
	};
	Vec2() : u(0), v(0) {}
	Vec2(t _u, t _v) : u(_u),v(_v) {}
	
	inline Vec2<t> operator +(const Vec2<t> &V) const { return Vec2<t>(u+V.u, v+V.v); }
	inline Vec2<t> operator -(const Vec2<t> &V) const { return Vec2<t>(u-V.u, v-V.v); }
	inline Vec2<t> operator *(float f)          const { return Vec2<t>(u*f, v*f); }
	inline t       dot  (const Vec2<t> &v) 		const { return x*v.x + y*v.y; }
	
	template <class > friend std::ostream& operator<<(std::ostream& s, Vec2<t>& v);
};

template <class t> struct Vec3 {
	union {
		struct {t x, y, z;};
		struct { t ivert, iuv, inorm; };
		t raw[3];
	};
	Vec3() : x(0), y(0), z(0) {}
	Vec3(t _x, t _y, t _z) : x(_x),y(_y),z(_z) {}
	
	inline Vec3<t> operator +(const Vec3<t> &v) const { return Vec3<t>(x+v.x, y+v.y, z+v.z); }
	inline Vec3<t> operator -(const Vec3<t> &v) const { return Vec3<t>(x-v.x, y-v.y, z-v.z); }
	inline Vec3<t> operator *(float f)          const { return Vec3<t>(x*f, y*f, z*f); }
	inline t       dot  (const Vec3<t> &v) 		const { return x*v.x + y*v.y + z*v.z; }
	inline Vec3<t> cross(const Vec3<t> &v) 		const { return Vec3<t>(y*v.z-z*v.y, z*v.x-x*v.z, x*v.y-y*v.x); }
	
	float 	  norm () const { return std::sqrt(x*x+y*y+z*z); }
	Vec3<t> & normalize(t l=1) { *this = (*this)*(l/norm()); return *this; }
	Vec4<t>   homogenize(bool is_point = true) { return Vec4<t>(x, y, z, is_point ? 1 : 0); }
	
	template <class > friend std::ostream& operator<<(std::ostream& s, Vec3<t>& v);
};

template <class t> struct Vec4 {
	union {
		struct {t x, y, z, w;};
		t raw[4];
	};
	Vec4() : x(0), y(0), z(0), w(0) {}
	Vec4(t _x, t _y, t _z, t _w) : x(_x),y(_y),z(_z),w(_w) {}
	
	inline Vec4<t> operator +(const Vec4<t> &v) const { return Vec4<t>(x+v.x, y+v.y, z+v.z, w+v.w); }
	inline Vec4<t> operator -(const Vec4<t> &v) const { return Vec4<t>(x-v.x, y-v.y, z-v.z, w-v.w); }
	inline Vec4<t> operator *(float f)          const { return Vec4<t>(x*f, y*f, z*f, w*f); }
	inline t       dot  (const Vec4<t> &v) 		const { return x*v.x + y*v.y + z*v.z + w*v.w; }

	Vec3<t> dehomogenize() { if (std::abs(w) < 1e-6) { return Vec3<t>(x, y, z); } return Vec3<t>(x/w, y/w, z/w); }
	
	template <class > friend std::ostream& operator<<(std::ostream& s, Vec3<t>& v);
};

typedef Vec2<float> Vec2f;
typedef Vec2<int>   Vec2i;
typedef Vec3<float> Vec3f;
typedef Vec3<int>   Vec3i;
typedef Vec4<float> Vec4f;
typedef Vec4<int>   Vec4i;

template <class t> std::ostream& operator<<(std::ostream& s, Vec2<t>& v) {
	s << "(" << v.x << ", " << v.y << ")\n";
	return s;
}

template <class t> std::ostream& operator<<(std::ostream& s, Vec3<t>& v) {
	s << "(" << v.x << ", " << v.y << ", " << v.z << ")\n";
	return s;
}

template <class t> std::ostream& operator<<(std::ostream& s, Vec4<t>& v) {
	s << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")\n";
	return s;
}

class Matrix4 {
public:
    std::array<std::array<float, 4>, 4> m;

	Matrix4();
    static Matrix4 identity();
	static Matrix4 projection(float c);

	Matrix4 operator *(const Matrix4& a);
    Vec4f   operator *(const Vec4f& v);
	
	friend std::ostream& operator<<(std::ostream& s, Matrix4& m);
};

class Matrix3 {
public:
    std::array<std::array<float, 3>, 3> m;

	Matrix3();
    static Matrix3 identity();

	Matrix4 homogenize();

	Matrix3 operator *(const Matrix3& a);
    Vec3f   operator *(const Vec3f& v);
	
	friend std::ostream& operator<<(std::ostream& s, Matrix3& m);
};

#endif //__GEOMETRY_H__