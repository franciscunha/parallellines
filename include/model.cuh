// This file is forked from an initial version by Dmitry V. Sokolov (https://github.com/ssloy) that was
// provided as a starting point for the tinyrenderer course (https://github.com/ssloy/tinyrenderer)

#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>
#include "geometry.cuh"
#include "tgaimage.cuh"

enum TextureType
{
	DIFFUSE,
	NORMAL_MAP,
	SPECULAR
};

class Model
{
private:
	int n_verts_;
	int n_uvs_;
	int n_normals_;
	int n_faces_;

	
	// all vectors are packed into a single array to make memory management easier
	// 0 to n_verts-1 -> verts
	// n_verts to n_verts+n_uvs-1 -> uvs
	// n_verts+n_uvs to n_verts+n_uvs+n_normals-1 -> normals 
	Vec3f *vectors_;

	// all indexes are packed into a single array to make memory management easier
	// 0 to 3*n_faces-1 -> face indexes
	// 3*n_faces to 6*n_faces-1 -> uv indexes
	// 6*n_faces to 9*n_faces-1 -> normal indexes
	int *indexes_;

	TGAImage diffuse_;
	TGAImage normal_map_;
	TGAImage specular_;

	__host__ __device__ TGAImage *texture_of_type(TextureType type);

public:
	Model(const char *filename);
	~Model();

	__host__ __device__ int nverts() { return n_verts_; }
	__host__ __device__ int nfaces() { return n_faces_; }
	
	__host__ __device__ int *face(int idx) { return &indexes_[idx * 3]; }
	__host__ __device__ int *face_uvs(int idx) { return &indexes_[3 * n_faces_ + (idx * 3)]; }
	__host__ __device__ int *face_normals(int idx) { return &indexes_[6 * n_faces_ + (idx * 3)]; }
	
	__host__ __device__ Vec3f vert(int i) { return vectors_[i]; }
	__host__ __device__ Vec2f uv(int i) { return Vec2f(vectors_[i + n_verts_].x, vectors_[i + n_verts_].y); }
	__host__ __device__ Vec3f normal(int i) { return vectors_[i + n_verts_ + n_uvs_]; }

	void load_texture(const char *filename, TextureType type);
	__host__ __device__ TGAColor sample_texture(Vec2f uv, TextureType type);
};

#endif //__MODEL_H__