#include "../include/line_renderer.hpp"

namespace line_renderer 
{
	namespace // private functions
	{
		Vec2i world_to_screen_coords(Vec3f world_coords, int width, int height)
		{
			return Vec2i(
				std::round((world_coords.x + 1.0f) * (float)width / 2.0f),
				std::round((world_coords.y + 1.0f) * (float)height / 2.0f));
		}
	}


	void draw(Vec2i p0, Vec2i p1, TGAImage &output, TGAColor color)
	{
		int x0 = p0.x, y0 = p0.y;
		int x1 = p1.x, y1 = p1.y;

		bool steep = false;
		if (std::abs(x0 - x1) < std::abs(y0 - y1))
		{
			steep = true;
			std::swap(x0, y0);
			std::swap(x1, y1);
		}

		if (x1 < x0)
		{
			std::swap(x0, x1);
			std::swap(y0, y1);
		}

		float slope = (y0 - y1) / (float)(x0 - x1);
		float y = y0;

		for (int x = x0; x <= x1; x++)
		{
			if (!steep)
			{
				output.set(x, std::round(y), color);
			}
			else
			{
				output.set(std::round(y), x, color);
			}
			y = y + slope;
		}
	}

	void wireframe(Model &model, TGAImage &output, TGAColor color)
	{
		int w = output.get_width();
		int h = output.get_height();

		for (int i = 0; i < model.nfaces(); i++)
		{
			int *face_vertices = model.face(i);

			for (int j = 0; j < 3; j++)
			{
				Vec2i vertex0 = world_to_screen_coords(model.vert(face_vertices[j]), w, h);
				Vec2i vertex1 = world_to_screen_coords(model.vert(face_vertices[(j + 1) % 3]), w, h);
				draw(vertex0, vertex1, output, color);
			}
		}
	}
}