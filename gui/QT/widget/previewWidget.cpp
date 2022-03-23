#include <stdio.h>
#include <QMessageBox>
#include <QOpenGLFunctions_3_3_Core>
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
#include <QOpenGLVersionFunctionsFactory>
#endif
#include "previewWidget.hpp"
#include "shared_mem_frame.hpp"

static const GLfloat rectangle[] = {
	 1.0f,  1.0f,  1.0f,  0.0f,
	-1.0f,  1.0f,  0.0f,  0.0f,
	-1.0f, -1.0f,  0.0f,  1.0f,

	 1.0f,  1.0f,  1.0f,  0.0f,
	-1.0f, -1.0f,  0.0f,  1.0f,
	 1.0f, -1.0f,  1.0f,  1.0f
};

static const char *vert_src = R"END(
#version 330 core
layout(location = 0) in vec2 vert_pos;
layout(location = 1) in vec2 vert_uv;

out vec2 UV;

uniform vec2 scale_vec;

void main(){
	gl_Position = vec4(vert_pos * scale_vec, 0.0f, 1.0f);
	UV = vert_uv;
}
)END";

static const char *frag_src = R"END(
#version 330 core
in vec2 UV;
out vec3 color;
uniform sampler2D tex;
void main(){
	color = texture(tex, UV).rgb;
}
)END";

static void compileShader(GLuint shaderId, QOpenGLFunctions_3_3_Core *f){
	f->glCompileShader(shaderId);

	GLint ret = GL_FALSE;
	int len;

	f->glGetShaderiv(shaderId, GL_COMPILE_STATUS, &ret);
	f->glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &len);
	if (len > 0){
		std::vector<char> errorMsg(len+1);
		f->glGetShaderInfoLog(shaderId, len, NULL, &errorMsg[0]);
		printf("%s\n", errorMsg.data());
	}
}


static unsigned char pixels[] = {
	255, 0, 0,   0, 255, 0,   0, 0, 255,   255, 255, 255,
	255, 0, 0,   0, 255, 0,   0, 0, 255,   255, 255, 255,
	255, 0, 0,   0, 255, 0,   0, 0, 255,   255, 255, 255,
	255, 0, 0,   0, 255, 0,   0, 0, 255,   255, 255, 255
};

static QOpenGLFunctions_3_3_Core *getOpenGLFuncs(){
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
	return QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_3_3_Core>();
#else
	return QOpenGLVersionFunctionsFactory::get<QOpenGLFunctions_3_3_Core>(QOpenGLContext::currentContext());
#endif
}

PreviewWidget::~PreviewWidget(){
	auto f = getOpenGLFuncs();

	makeCurrent();
	f->glDeleteBuffers(1, &vertexBuffer);
	f->glDeleteProgram(program);
	f->glDeleteTextures(1, &texture);
	vao.destroy();
	doneCurrent();
}

void PreviewWidget::initializeGL(){
	auto f = getOpenGLFuncs();

	if(!f) {
		QMessageBox warningBox(this);
		warningBox.setText("OpenGL 3.3 is required!");
		warningBox.exec();
		exit(1);
	}

	vao.create();
	vao.bind();

	f->glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	f->glDisable(GL_DEPTH_TEST);
	f->glDisable(GL_SCISSOR_TEST);
	f->glDisable(GL_STENCIL_TEST);

	f->glGenBuffers(1, &vertexBuffer);
	f->glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	f->glBufferData(GL_ARRAY_BUFFER, sizeof(rectangle), rectangle, GL_STATIC_DRAW);

	GLuint vertexShader = f->glCreateShader(GL_VERTEX_SHADER);
	GLuint fragShader = f->glCreateShader(GL_FRAGMENT_SHADER);

	f->glShaderSource(vertexShader, 1, &vert_src, NULL);
	compileShader(vertexShader, f);
	f->glShaderSource(fragShader, 1, &frag_src, NULL);
	compileShader(fragShader, f);

	program = f->glCreateProgram();
	f->glAttachShader(program, vertexShader);
	f->glAttachShader(program, fragShader);
	f->glLinkProgram(program);
	f->glUseProgram(program);

	f->glDetachShader(program, vertexShader);
	f->glDetachShader(program, fragShader);
	f->glDeleteShader(vertexShader);
	f->glDeleteShader(fragShader);

	f->glGenTextures(1, &texture);
	f->glBindTexture(GL_TEXTURE_2D, texture);
	f->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 4, 4, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	vao.release();

	scaleVec[0] = 0.75f;
	scaleVec[1] = 0.5;

	vidW = 1280;
	vidH = 720;
}

void PreviewWidget::calculateScale(){
	double videoAspect = (double) vidW / vidH;
	double widgetAspect = (double) width / height;

	if(videoAspect > widgetAspect){
		float scale = widgetAspect / videoAspect;
		scaleVec[0] = 1;
		scaleVec[1] = scale;
	} else {
		float scale = videoAspect / widgetAspect;
		scaleVec[0] = scale;
		scaleVec[1] = 1;
	}
}

void PreviewWidget::resizeGL(int w, int h){
	width = w;
	height = h;

	calculateScale();
}

void PreviewWidget::setVidSize(int w, int h){
	if(vidW == w && vidH == h)
		return;

	vidW = w;
	vidH = h;
	calculateScale();
}

void PreviewWidget::paintGL(){
	auto f = getOpenGLFuncs();

	if(!f)
		return;

	vao.bind();
	f->glClear(GL_COLOR_BUFFER_BIT);

	f->glUseProgram(program);

	f->glEnableVertexAttribArray(0);
	f->glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	f->glVertexAttribPointer(
			0,
			2,
			GL_FLOAT,
			GL_FALSE,
			4 * sizeof(float),
			(void*)0
			);
	f->glEnableVertexAttribArray(1);
	f->glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	f->glVertexAttribPointer(
			1,
			2,
			GL_FLOAT,
			GL_FALSE,
			4 * sizeof(float),
			(void*)(2 * sizeof(float))
			);

	GLuint loc;
	loc = f->glGetUniformLocation(program, "scale_vec");
	f->glUniform2fv(loc, 1, scaleVec);

	f->glBindTexture(GL_TEXTURE_2D, texture);
	struct Shared_mem_frame *sframe = shared_mem.get_frame_and_lock();
	if(sframe){
		f->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, sframe->width, sframe->height, 0, GL_RGB, GL_UNSIGNED_BYTE, sframe->pixels);
		setVidSize(sframe->width, sframe->height);
		shared_mem.unlock();
		//Detach to prevent deadlocks
		shared_mem.detach();
	} else {
		f->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 4, 4, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	}

	f->glDrawArrays(GL_TRIANGLES, 0, 6);

	vao.release();
}

void PreviewWidget::setKey(const char *key){
	shared_mem.detach();
	shared_mem.setKey(key);
}

void PreviewWidget::start(){
	timer.start(1000/24);
}

void PreviewWidget::stop(){
	timer.stop();
}
