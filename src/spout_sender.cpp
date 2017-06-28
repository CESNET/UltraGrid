#include "SpoutSDK/Spout.h"

#include "spout_sender.h"

void *spout_sender_register(const char *name, int width, int height) {

	SpoutSender *sender = new SpoutSender;
	sender->CreateSender(name, width, height);

	return sender;
}

void spout_sender_sendframe(void *s, int width, int height, unsigned int id) {
	((SpoutSender *)s)->SendTexture(id, GL_TEXTURE_2D, width, height, false); // default is flip
	}

void spout_sender_unregister(void *s) {
        ((SpoutSender *)s)->ReleaseSender();
        delete s;
}

