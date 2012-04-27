//
//  UltraGrid_GUIAppDelegate.h
//  UltraGrid GUI
//
//  Created by Martin Pulec on 10/31/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import "Settings.h"
#import "Terminal.h"
#import "UGController.h"

@interface UltraGrid_GUIAppDelegate : NSObject <NSApplicationDelegate> {
    NSWindow *window;
	NSWindow *advancedWin;
	Settings *values;
	
	NSString *execPath;
	
	
	UGController *controller;
	NSFileHandle *outputHandle;
}

@property (assign) IBOutlet NSWindow *window;
@property (assign) IBOutlet NSWindow *advancedWin;
@property (assign) IBOutlet Settings *settings;
@property (assign) IBOutlet Terminal *terminal;
@property (assign) IBOutlet NSString *remoteAddress;

@property (assign) IBOutlet NSButtonCell *startButton;

//@property (assign) IBOutlet NSMutableArray *displays;
-(void) doPlay;
-(void) outputReady: (id) sender;
-(void) terminated: (id) sender;
-(NSString *) getCompressionString;

@end
