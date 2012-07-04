//
//  main.m
//  UltraGrid GUI
//
//  Created by Martin Pulec on 10/31/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <Cocoa/Cocoa.h>

int main(int argc, char *argv[])
{
	NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
	NSString *execPath = [[NSBundle mainBundle] pathForAuxiliaryExecutable:@"uv"];
	if(!execPath) {
		const char *header = "No UltraGrid binary";
		const char *message = "Cannot found UltraGrid binary.\n"
					"Please, copy the binary inside the bundle "
					"as indicated in README (ultragrid/gui/UltraGrid GUI/README)";
		CFStringRef header_ref = CFStringCreateWithCString(NULL,header,strlen(header));
		CFStringRef message_ref = CFStringCreateWithCString(NULL,message,strlen(message));

		CFOptionFlags result;

		CFUserNotificationDisplayAlert(
				0,
				kCFUserNotificationNoteAlertLevel,
				NULL,
				NULL,
				NULL,
				header_ref,
				message_ref,
				NULL,
				NULL,
				NULL,
				&result
				);

		CFRelease(header_ref);
		CFRelease(message_ref);
		[pool release];
		return 1;
	}
	// we do not need the pool no more
	[pool release];

	return NSApplicationMain(argc,  (const char **) argv);
}
