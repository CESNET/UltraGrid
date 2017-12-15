#ifndef PREVIEWWIDGET_HPP
#define PREVIEWWIDGET_HPP

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QSharedMemory>
#include <QTimer>

class PreviewWidget : public QOpenGLWidget{
public:
	PreviewWidget(QWidget *parent) : QOpenGLWidget(parent) {
		connect(&timer, SIGNAL(timeout()), this, SLOT(update()));
		timer.start(1000/24);
  	}

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

	GLfloat scaleVec[2];
	int vidW, vidH;
	int width, height;

	void setVidSize(int w, int h);
	void calculateScale();

	QSharedMemory shared_mem;
	QTimer timer;
};

#endif
