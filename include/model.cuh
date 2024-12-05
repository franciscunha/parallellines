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
	int nverts_;
	int nfaces_;

	Vec3f *verts_;
	Vec2f *uvs_;
	Vec3f *normals_;

	int *faces_;
	int *faces_uvs_;
	int *faces_normals_;

	TGAImage diffuse_;
	TGAImage normal_map_;
	TGAImage specular_;

	__host__ __device__ TGAImage *texture_of_type(TextureType type);

public:
	Model(const char *filename);
	~Model();
	
	__host__ __device__ int Model::nverts() { return nverts_; }
	__host__ __device__ int Model::nfaces() { return nfaces_; }
	
	__host__ __device__ int *Model::face(int idx) { return &faces_[idx * 3]; }
	__host__ __device__ int *Model::face_uvs(int idx) { return &faces_uvs_[idx * 3]; }
	__host__ __device__ int *Model::face_normals(int idx) { return &faces_normals_[idx * 3]; }
	
	__host__ __device__ Vec3f Model::vert(int i) { return verts_[i]; }
	__host__ __device__ Vec2f Model::uv(int i) { return uvs_[i]; }
	__host__ __device__ Vec3f Model::normal(int i) { return normals_[i]; }

	void load_texture(const char *filename, TextureType type);
	__host__ __device__ TGAColor sample_texture(Vec2f uv, TextureType type);
};

#endif //__MODEL_H__