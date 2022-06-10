#ifndef DEBUG_HPP_db8549e3cb58
#define DEBUG_HPP_db8549e3cb58

#include <cassert>
#include <QMessageBox>
#include <stdlib.h>

#define ASSERT_GUI(x) if(!(x)) abort_msgBox("Assert failed: " #x)

#ifdef DEBUG
#	define DEBUG_ASSERT(x) assert(x)
#	define DEBUG_ASSERT_GUI(x) ASSERT_GUI(x)
#else
#	define DEBUG_ASSERT(x) do{  }while(0)
#	define DEBUG_ASSERT_GUI(x) do{  }while(0)
#endif

#define DEBUG_FAIL(x) DEBUG_ASSERT((x) && false)

static inline void abort_msgBox(const char *msg){
	QMessageBox msgBox;
	msgBox.setText(msg);
	msgBox.setIcon(QMessageBox::Critical);
	msgBox.exec();
	exit(EXIT_FAILURE);
}

#endif //DEBUG_HPP_db8549e3cb58
