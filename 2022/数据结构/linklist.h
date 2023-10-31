#pragma once
class P {
private:int ID;
	   char name[10];
	   char ch;
	   char phone[13];
	   char addr[31];
public:
	void setP() {
	cout << "set ID" << endl;
	cin >> ID;
	cout << "set name" << endl;
	cin >> name[10];
	cout << "set sex,'b'for boy,'g'for girl" << endl;
	cin >> ch;
	cout << "set phone" << endl;
	cin >> phone[13];
	cout << "set address" << endl;
	cin >> addr[31];
}
	  void changeID() {
		  cout << "put new ID" << endl;
		  int x;
		  cin >> x;
		  cout << "sure?'1'for yes,'2'for no" << endl;
		  int t;
		  cin >> t;
		  if (t)
			  ID = x;
	  }
	  void changename() {
		  cout << "put new name" << endl;
		  char* p = new char[10];
		  cin >> *p;
		  cout << "sure?'1'for yes,'2'for no" << endl;
		  int t;
		  cin >> t;
		  if (t)
			  name[10] = *p;
		  delete[]p;
	  }
	  void changesex() {
		  cout << "put new sex,'b'for boy,'g'for girl" << endl;
		  char x;
		  cin >> x;
		  cout << "sure?'1'for yes,'2'for no" << endl;
		  int t;
		  cin >> t;
		  if (t) {
			  if (ch != x)
				  ch = x;
			  else cout << "you don't have to change" << endl;
		  }
	  }
	  void changephone() {
		  cout << "put new phone" << endl;
		  char* p = new char[13];
		  cin >> *p;
		  cout << "sure?'1'for yes,'2'for no" << endl;
		  int t;
		  cin >> t;
		  if (t)
			  phone[13] = *p;
		  delete[]p;
	  }
	  void changeaddress() {
		  cout << "put new address" << endl;
		  char* p = new char[31];
		  cin >> *p;
		  cout << "sure?'1'for yes,'2'for no" << endl;
		  int t;
		  cin >> t;
		  if (t)
			  addr[31] = *p;
		  delete[]p;
	  }
	  int GetID() {
		  int getID;
		  getID = ID;
		  return getID;
	  }
	  char Getname() {
		  const char* p = &name[10];
		  return *p;
	  }
	  char Getsex() {
		  char p = ch;
		  return p;
	  }
	  char Getphone() {
		  const char* p = &phone[13];
		  return *p;
	  }
	  char Getaddress() {
		  const char* p = &addr[31];
		  return *p;
	  }
	  ~P() {
		  ID = 0;
		  name[10] = 0;
		  ch = 0;
		  phone[13] = 0;
		  addr[31] = 0;
	  }
	  void deleteP() {
		  cout << "sure?'1'for yes,'2'for no" << endl;
		  int t;
		  cin >> t;
		  if (t)
			  P::~P();
	  }
}
template <class T> struct Node
{
    T data;
    Node<T>* next;
	Node<T>* prior;
};

template <class T> class LinkList
{
public:
    LinkList();
    LinkList(T a[], int n);
    int 	Length();
    Node<T>* Get(int);                        //查找
    int        Locate(T);                      //定位
   void 	Insert(int i, T x);                  //插入
    T	Delete(int);                    //删除
    ~LinkList();
    void print();
private:
    Node<T>* front;
};
