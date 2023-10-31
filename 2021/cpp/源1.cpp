#include<iostream>

void main() {
	int a = 2, b = 13, c = 24, d = 46, e = 57, f = 60, g = 79, h = 83;
	std::cout << "请输入一个数" << std::endl;
	int i = 0;
	std::cin >> i;
	if (i <= a)
		std::cout << i << " " << a << " " << b << " " << c << " " << d << " " << e << " " << f << " " << g << " " << h << std::endl;
	else
	{
		if (i <= b)
			std::cout << a << " " << i << " " << b << " " << c << " " << d << " " << e << " " << f << " " << g << " " << h << std::endl;
		else {
			if(i<=c)
				std::cout << a << " " << b << " " << i << " " << c << " " << d << " " << e << " " << f << " " << g << " " << h << std::endl;
			else {
				if(i<=d)
					std::cout << a << " " << b << " " << c << " " << i << " " << d << " " << e << " " << f << " " << g << " " << h << std::endl;
				else {
					if(i<=e)
						std::cout << a << " " << b << " " << c << " " << d << " " << i << " " << e << " " << f << " " << g << " " << h << std::endl;
					else {
						if(i<=f)
							std::cout << a << " " << b << " " << c << " " << d << " " << e << " " << i << " " << f << " " << g << " " << h << std::endl;
						else {
							if(i<=g)
								std::cout << a << " " << b << " " << c << " " << d << " " << e << " " << f << " " << i << " " << g << " " << h << std::endl;
							else {
								if(i<=h)
									std::cout << a << " " << b << " " << c << " " << d << " " << e << " " << f << " " << g << " " << i << " " << h << std::endl;
								else
									std::cout << a << " " << b << " " << c << " " << d << " " << e << " " << f << " " << g << " " << h << " " << i << std::endl;

							}
						}
					}
				}
			}
		}
	}
}