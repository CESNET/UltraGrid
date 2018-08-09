#ifndef PREVIEWWIDGET_HPP
#define PREVIEWWIDGET_HPP

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QSharedMemory>
#include <QOpenGLVertexArrayObject>
#include <QTimer>

#include "shared_mem_frame.hpp"

class PreviewWidget : public QOpenGLWidget{
public:
	PreviewWidget(QWidget *parent) : QOpenGLWidget(parent) {
		connect(&timer, SIGNAL(timeout()), this, SLOT(update()));
  	}

	void setKey(const char *key);
	void start();
	void stop();

protected:
	void initializeGL();
	void resizeGL(int w, int h);
	void paintGL();

private:
	GLuint vertexBuffer;
	GLuint vertexShader;
	GLuint fragShader;
	GLuint program;
	GLuint texture;

	QOpenGLVertexArrayObject vao;

	GLfloat scaleVec[2];
	int vidW, vidH;
	int width, height;

	void setVidSize(int w, int h);
	void calculateScale();

	Shared_mem shared_mem;
	QTimer timer;
};

#endif
