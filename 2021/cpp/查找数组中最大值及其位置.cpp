#include<iostream>
using namespace std;
void search(int* pa, int n, int* pmax, int&y) {
	int i = 1;
	for(i=1;i<n;i++)
		if (*pmax < pa[i]) {
			*pmax = pa[i];
			y = i+1;
		}
}

int main() {
	int a[10];
	int i = 0;
	cout << "请输入十个数" << endl;
	for (i = 0; i < 10; i++)
		cin >> a[i];
	int* pa = a;
	int* pmax =&a[0];
	int y ;
	search(pa, 10, pmax, y);
	cout << *pmax << " " << y;
	return 0;
}