#include "SpoutSDK/Spout.h"

#include "spout_receiver.h"

void *spout_create_receiver(char *name, unsigned int *width, unsigned int *height) {
	SpoutReceiver *receiver = new SpoutReceiver;
	receiver->CreateReceiver(name, *width, *height);
        bool connected;
        receiver->CheckReceiver(name, *width, *height, connected);
        if (!connected) {
                fprintf(stderr, "[SPOUT] Not connected to server '%s'. Is it running?\n", name);
                receiver->ReleaseReceiver();
                delete receiver;
                return NULL;
        }

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

