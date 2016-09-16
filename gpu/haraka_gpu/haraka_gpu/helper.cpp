
#include "helper.h"

#include <iomanip>

void printVector(const string& message, const vector<char>& vec_to_print)
{
	cout << endl;
	if (!message.empty())
		cout << message << endl;

	if (vec_to_print.empty())
		return;

	cout << "Size of Vector: " << vec_to_print.size() << " byte" << endl;

	for (unsigned int i = 0; i < vec_to_print.size(); ++i)
	{
		printf("%02x ", (unsigned char)vec_to_print[i]);

		if (!((i + 1) % 4))
			cout << "  ";

		if (!((i + 1) % (vec_to_print.size() / 4)))
			cout << endl;
	}
}