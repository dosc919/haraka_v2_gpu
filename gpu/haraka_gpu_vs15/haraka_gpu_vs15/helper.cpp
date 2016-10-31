
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

	for (unsigned int i = 0; i < 4; ++i)
	{
		for (unsigned int j = i; j < vec_to_print.size(); j += 4)
		{
			printf("%02x ", (unsigned char)vec_to_print[j]);

			if (!(((j - i) / 4 + 1) % 4))
				cout << "  ";
		}
		cout << endl;
	}
}