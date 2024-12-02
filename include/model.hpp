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
	std::vector<Vec3f> verts_;
	std::vector<Vec2f> uvs_;
	std::vector<Vec3f> normals_;
	std::vector<std::vector<int>> faces_;
	std::vector<std::vector<int>> faces_uvs_;
	std::vector<std::vector<int>> faces_normals_;
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
	std::vector<int> face(int idx);
	std::vector<int> face_uvs(int idx);
	std::vector<int> face_normals(int idx);

	void load_texture(const char *filename, TextureType type);
	TGAColor sample_texture(Vec2f uv, TextureType type);
};

#endif //__MODEL_H__