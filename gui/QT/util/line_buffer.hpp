#ifndef LINE_BUFFER_5dd68dbd1e7b
#define LINE_BUFFER_5dd68dbd1e7b

#include <string>
#include <string_view>

class LineBuffer{
public:
	void write(std::string_view str);

	bool hasLine() const { return startIdx != endIdx; }

	//valid only until pop()
	std::string_view peek() const {
		return std::string_view(buf.data() + startIdx, endIdx - startIdx);
	}

	void pop();

private:
	void reduceBuf();

	std::string buf;
	size_t startIdx = 0;
	size_t endIdx = 0;
};


#endif
