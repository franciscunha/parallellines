// This file is forked from an initial version by Dmitry V. Sokolov (https://github.com/ssloy) that was
// provided as a starting point for the tinyrenderer course (https://github.com/ssloy/tinyrenderer)

#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <fstream>

#pragma pack(push, 1)
struct TGA_Header
{
	char idlength;
	char colormaptype;
	char datatypecode;
	short colormaporigin;
	short colormaplength;
	char colormapdepth;
	short x_origin;
	short y_origin;
	short width;
	short height;
	char bitsperpixel;
	char imagedescriptor;
};
#pragma pack(pop)

struct TGAColor
{
	union
	{
		struct
		{
			unsigned char b, g, r, a;
		};
		unsigned char raw[4];
		unsigned int val;
	};
	int bytespp;

	__host__ __device__ TGAColor() : val(0), bytespp(1)
	{
	}

	__host__ __device__ TGAColor(unsigned char R, unsigned char G, unsigned char B, unsigned char A) : b(B), g(G), r(R), a(A), bytespp(4)
	{
	}

	__host__ __device__ TGAColor(int v, int bpp) : val(v), bytespp(bpp)
	{
	}

	__host__ __device__ TGAColor(const TGAColor &c) : val(c.val), bytespp(c.bytespp)
	{
	}

	__host__ __device__ TGAColor(const unsigned char *p, int bpp) : val(0), bytespp(bpp)
	{
		for (int i = 0; i < bpp; i++)
		{
			raw[i] = p[i];
		}
	}

	__host__ __device__ TGAColor &operator=(const TGAColor &c)
	{
		if (this != &c)
		{
			bytespp = c.bytespp;
			val = c.val;
		}
		return *this;
	}

	__host__ __device__ TGAColor operator*(float x) const
	{
		return TGAColor(this->r * x, this->g * x, this->b * x, this->a);
	};
};

class TGAImage
{
protected:
	unsigned char *data;
	int width;
	int height;
	int bytespp;

	bool load_rle_data(std::ifstream &in);
	bool unload_rle_data(std::ofstream &out);

public:
	enum Format
	{
		GRAYSCALE = 1,
		RGB = 3,
		RGBA = 4
	};

	TGAImage();
	TGAImage(int w, int h, int bpp);
	TGAImage(const TGAImage &img);
	TGAImage *cudaDeepCopyToDevice();
	void cudaDeepCopyFromDevice(const TGAImage &device_img);
	static void cudaDeepFree(TGAImage *device_ptr);
	bool read_tga_file(const char *filename);
	bool write_tga_file(const char *filename, bool rle = true);
	bool write_tga_file(std::string filename, bool rle = true);
	bool flip_horizontally();
	bool flip_vertically();
	bool scale(int w, int h);
	__host__ __device__ TGAColor get(int x, int y);
	__host__ __device__ bool set(int x, int y, TGAColor c);
	~TGAImage();
	TGAImage &operator=(const TGAImage &img);
	__host__ __device__ int get_width();
	__host__ __device__ int get_height();
	int get_bytespp();
	__host__ __device__ unsigned char *buffer();
	void clear();
};

#endif //__IMAGE_H__