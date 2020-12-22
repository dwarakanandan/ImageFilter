#include "image.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

void Image::CopyImage(const Image &source)
{
	SetWidth(source.GetWidth());
	SetHeight(source.GetHeight());
	SetComponents(source.GetComponents());
	for (unsigned int y = 0; y < GetHeight(); y++)
	{
		for (unsigned int x = 0; x < GetWidth(); x++)
		{
			for (unsigned int c = 0; c < GetComponents(); c++)
			{
				unsigned int v = source.VGetValue(x, y, c);
				v *= GetMask();
				v /= source.GetMask();
				VSetValue(x, y, c, v);
			}
		}
	}
}
