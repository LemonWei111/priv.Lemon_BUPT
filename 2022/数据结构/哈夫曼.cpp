#include<iostream>
#include<cstring>
#include<stdio.h>
#include <vector>
#include <algorithm>
using namespace std;
struct HNode  //哈夫曼结点 
{
	int weight;
	int parent;
	int LChild;
	int RChild;
};
struct HCode  //编码表
{
	char data;
	char code[100];
};
class Huffman //哈夫曼类
{
private:
	HNode* HTree;
	HCode* HCodetable;
	char str[1024] = { 0 };  //原始字符串
	char leaf[256] = { 0 };   //叶子结点对应字符
	int amount[256] = { 0 };   //字符出现次数
public:
	int n=0;          //叶子结点数
	void putin();  //写入原字符串并记录出现过的字符
	void getmin(HNode*, int, int& , int& );
	void createHTree();
	void Init();
	void CreateTable();
	void Encoding();
	void Decoding();
	void printHTree();
	~Huffman();
};
void Huffman::putin() 
{
	int a[256] = { 0 };
	int x = cin.get();
	int i = 0;
	while ((x != '\n') && (x != '\r')) 
	{
		a[x]++;
		str[i++] = x;
		x = cin.get();
	}
	str[i] = '\0';
	n = 0;
	for (i = 0; i < 256; i++) 
	{
		if (a[i] > 0)  //判断字符是否出现（=0说明未出现） 
		{
			leaf[n] = (char)i;
			amount[n] = a[i];
			n++;
		}
	}
}
void Huffman::getmin(HNode*Htree, int n, int &lf, int&ls) 
{
	int j;
	//找一个比较值的起始值
	for (j = 0; j < 2*n-1; j++) //找lf
	{
		if (Htree[j].parent == -1)
		{
			lf = j;      break;
		}
	}
	j++;
	for (; j < 2*n-1; j++) //找ls
	{
		if (Htree[j].parent == -1)
		{
			ls = j;           break;
		}
	}
	if (Htree[lf].weight > Htree[ls].weight) //lf指向最小的
	{
		int x = ls;           ls = lf;          lf = x;
	}
	j++;
	for (; j < 2*n-1; j++)
	{
		if (Htree[j].parent == -1)
		{
			if (Htree[j].weight < Htree[lf].weight)
			{
				ls =lf;              lf = j;
			}
			else if (Htree[j].weight < Htree[ls].weight)
			{
				ls = j;
			}
		}
	}
}
void Huffman::createHTree() 
{
	HNode* HTree = new HNode[2 * n - 1];
	for(int i=0;i<2*n-1;i++)
	{
		HTree[i].weight = amount[i];
		HTree[i].LChild = -1;
		HTree[i].RChild = -1;
		HTree[i].parent = -1;
	}
	int lf, ls;    //最小两权重的下标
	for (int i = 0; i < 2*n-1; i++) 
	{
		getmin(HTree,n,lf, ls);
		HTree[lf].parent = HTree[ls].parent = i;
		HTree[i].weight= HTree[lf].weight +HTree[ls].weight ;
		HTree[i].LChild = lf;
		HTree[i].RChild = ls;
		HTree[i].parent = -1;
	}
}
void Huffman::Init() 
{
	cout << "请输入需要编码的内容" << endl;
	putin();
	createHTree();
}
void Huffman::CreateTable() 
{
	HCode* HCodetable = new HCode[n];
	cout << "正在创建编码表" << endl;
	for (int i = 0; i < n; i++) 
	{
		HCodetable[i].data = str[i];
		int i1 = i;
		int i2 = HTree[i].parent;
		int k = 0;
		while (i2 != -1) 
		{
			if (i1 == HTree[i2].LChild)
				HCodetable[i].code[k] = '0';
			else
				HCodetable[i].code[k] = '1';
			k++;
			i1 = i2;
			i2 = HTree[i1].parent;
		}
		HCodetable[i].code[k] = '\0';
		int tmp = 0;
		char* s = new char[1000];
		for (int j = strlen(HCodetable[i].code)-1; j>=0; j--)
		{
			tmp = HCodetable[i].code[j];
			for(int k=0;k < strlen(HCodetable[i].code);k++)
			s[k] = tmp;
		}
		for (int j = 0; j < strlen(HCodetable[i].code); j++)
		{
			HCodetable[i].code[j] = s[j];
			cout <<"'"<<HCodetable[i].data<<"' " << HCodetable[i].code[j] << endl;
		}
		delete[]s;
	}
}
void Huffman::Encoding() 
{
	cout << "正在编码" << endl;
	char* s = str;
	char* am=new char[1000];
	while (*s != '\0') 
	{
		for(int i=0;i<strlen(str);i++)
			if (*s == HCodetable[i].data) 
			{
				for (int j = 0; j < strlen(str); j++)
					am[j] = HCodetable[i].code[j];
				break;
			}
		s++;
	}
	for(int i=0;i< strlen(str);i++)
		str[i] = am[i];
	cout <<"编码结果："<<endl<< str << endl;
	delete[]am;
}
void Huffman::Decoding() 
{
	cout << "正在解码" << endl;
	char* s=str;
	char* d=new char[1000];
	while (*s != '\0')
	{
		int k = 2 * n - 2;
		while (HTree[k].LChild != -1) 
		{
			if (*s == '0')
				k = HTree[k].LChild;
			else
				k = HTree[k].RChild;
			s++;
		}
		*d= HCodetable[k].data;
		d++;
	}
	cout << *d;
	delete[]d;
}
void Huffman::printHTree()
{
	for (int i = 0; i < 2 * n - 1; i++)
	{
		cout << amount[i] << " ";
		cout << HTree[i].weight << "  " << HTree[i].parent << endl;
	}
}
Huffman::~Huffman() 
{
	cout << "正在退出" << endl;
	delete[]HTree;
	delete[]HCodetable;
	cout << "已退出" << endl;
}	
int main() 
{
	Huffman huffman;
	huffman.Init();
	huffman.createHTree();
	huffman.printHTree();
	huffman.CreateTable();
	huffman.Encoding();
	cout << "是否查看解码结果? 1 yes,0 no" << endl;
	int x=0;
	if (x)
		huffman.Decoding();
	else;
	return 0;
}
