#include <cassert>
#include "line_buffer.hpp"

void LineBuffer::write(std::string_view str){
	reduceBuf();

	buf += str;

	if(!hasLine()){
		pop();
	}
}

void LineBuffer::pop(){
	startIdx = endIdx;
	auto idx = buf.find('\n', endIdx);
	if(idx != buf.npos){
		endIdx = idx + 1;
	}
}

void LineBuffer::reduceBuf(){
	assert(startIdx <= endIdx);
	buf.erase(0, startIdx);
	endIdx = endIdx - startIdx;
	startIdx = 0;
}
