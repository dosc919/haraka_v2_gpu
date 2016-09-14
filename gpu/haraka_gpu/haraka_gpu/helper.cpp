
#include "helper.h"

#include <iomanip>

void printVector(const string& message, const vector<unsigned char>& vec_to_print)
{
	cout << endl;
	if (!message.empty())
		cout << message << endl;

	if (vec_to_print.empty())
		return;

	for (int i = 0; i < vec_to_print.size(); ++i)
	{
		cout << setw(2) << setfill('0') << hex << (int)vec_to_print[i] << " ";

		if (!((i + 1) % 4))
			cout << "  ";

		if (!((i + 1) % 16))
			cout << endl;
	}
}