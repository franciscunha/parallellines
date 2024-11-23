// This file is forked from an initial version by Dmitry V. Sokolov (https://github.com/ssloy) that was
// provided as a starting point for the tinyrenderer course (https://github.com/ssloy/tinyrenderer)

#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>
#include "geometry.hpp"
#include "tgaimage.hpp"

class Model {
private:
	std::vector<Vec3f> verts_;
	std::vector<Vec2f> uvs_;
	std::vector<std::vector<int> > faces_;
	std::vector<std::vector<int> > faces_uvs_;
	TGAImage texture_;
public:
	Model(const char *filename);
	~Model();
	int nverts();
	int nfaces();
	Vec3f vert(int i);
	Vec2f uv(int i);
	std::vector<int> face(int idx);
	std::vector<int> face_uvs(int idx);
	void load_texture(const char *filename);
	TGAColor sample_texture(Vec2f uv);
};

#endif //__MODEL_H__