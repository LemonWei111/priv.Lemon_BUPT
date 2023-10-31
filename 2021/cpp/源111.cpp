#include<iostream>
using namespace std;


char s[1000000];

int main() {
	cout << "请输入密码及密匙" << endl;
	int m = 0; char t = getchar();
	for (; !isdigit(t); t = getchar())
		s[++m] = t;
	
	for (int i = 1; i <= m; ++i) {
		if (!isalpha(s[i])) continue;
		if (s[i] - 'A' + 1 <= (t - '0'))
			s[i] += 26;
		s[i] -= (t - '0');
		
}
	printf("%s", s + 1);
	return 0;
}


/*int main() {
	int n = 0;
	cout << "请输入密匙" << endl;
	cin >> n;
	char sm;
	cout << "请输入密码" << endl;
	cout << "解码结果如下" << endl; 
	for (int i = 1; i >0; i++)
	{
		char s = getchar();//每一次输入始终会把输入的n提取为s并进行相同码值转换，所以此处应该考虑两次for循环
		sm =((s - 'a') - n )%26+ 'a';//对特殊字符也提取了ASCII码，所以在第二次用for循环时还应设条件
		cout << sm;
	}
	cin.get();
		return 0;
}*/