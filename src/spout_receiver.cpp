#include "SpoutSDK/Spout.h"

#include "spout_receiver.h"

void *spout_create_receiver(char *name, unsigned int *width, unsigned int *height) {
	SpoutReceiver *receiver = new SpoutReceiver;
	receiver->CreateReceiver(name, *width, *height);

	return receiver;
}

bool spout_receiver_recvframe(void *s, char *sender_name, unsigned int width, unsigned int height, char *data, GLenum glFormat) {
	return ((SpoutReceiver *)s)->ReceiveImage(sender_name, width, height, (unsigned char *) data, glFormat);
}

void spout_receiver_delete(void *s) {
        if (!s) {
                return;
        }
        ((SpoutReceiver *)s)->ReleaseReceiver();
        delete s;
}

