// This file is forked from an initial version by Dmitry V. Sokolov (https://github.com/ssloy) that was
// provided as a starting point for the tinyrenderer course (https://github.com/ssloy/tinyrenderer)

#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>
#include "geometry.hpp"
#include "tgaimage.hpp"

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

	TGAImage *texture_of_type(TextureType type);

public:
	Model(const char *filename);
	~Model();

	int nverts();
	int nfaces();

	Vec3f vert(int i);
	Vec2f uv(int i);
	Vec3f normal(int i);
	int *face(int idx);
	int *face_uvs(int idx);
	int *face_normals(int idx);

	void load_texture(const char *filename, TextureType type);
	TGAColor sample_texture(Vec2f uv, TextureType type);
};

#endif //__MODEL_H__