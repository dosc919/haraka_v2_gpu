
#include "helper.h"

#include <iomanip>

void printVector(const string& message, const vector<char>& vec_to_print)
{
	cout << endl;
	if (!message.empty())
		cout << message << endl;

	if (vec_to_print.empty())
		return;

	for (unsigned int i = 0; i < vec_to_print.size(); ++i)
	{
		cout << setw(2) << setfill('0') << hex << (int)vec_to_print[i] << " ";

		if (!((i + 1) % 4))
			cout << "  ";

		if (!((i + 1) % (vec_to_print.size() / 4)))
			cout << endl;
	}
}