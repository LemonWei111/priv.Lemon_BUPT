#include<iostream>
using namespace std;
void swap(int *x, int* y) {
	int t = 0;
	t = *x; *x = *y; *y = t;
}
void sort(int* arr, int n) {
	int i = 0; int m = n - 1;
	while (m) {
		int x = m;
		m = 0;
		for (i = 0; i < x; i++)
			if (arr[i] < arr[i + 1]) { 
				swap(arr[i], arr[i + 1]); 
		m = i; }
	}
}
void search(int* arr, int n, int x, int& y) {
	int i = 0;
	for (i = 0; (arr[i] >= x)&(i<n); i++);
	y = arr[i]+1;
}
int main() {
	int n,*p=0;
	cout << "请输入您要输入的数字个数" << endl;
	cin >> n;
	p = new int(n);
	int* a = p;
	int i = 0;
	cout << "请输入数字" << endl;
	for(i=0;i<n;i++)
	cin >> p[i];
	sort(a, n);
	cout << "排序后的数字为";
	for (i = 0; i < n; i++)
		cout << a[i] << " ";
	cout <<endl<< "请输入待插入数字";
	int c, y;
	cin >> c;
	search(a, n, c, y);
	cout << "应该插入的位置为" << y;
	a = 0;
	delete p;
	return 0;
}