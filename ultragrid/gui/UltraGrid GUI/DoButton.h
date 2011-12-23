//
//  DoButton.h
//  UltraGrid GUI
//
//  Created by Martin Pulec on 11/2/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import "UltraGrid_GUIAppDelegate.h"
#import "Terminal.h"

@interface DoButton : NSButton {
	NSWindow *advancedWin;
	UltraGrid_GUIAppDelegate *myApp;
}
- (IBAction)doPlay:(id)pId;

- (IBAction)doAdvanced:(id)pId;
- (IBAction)doAdvancedOK:(id)pId;
- (IBAction)doAdvancedSave:(id)pId;

- (IBAction)doCaptureHelp:(id)pId;
- (IBAction)doDisplayHelp:(id)pId;

@property (assign) IBOutlet NSWindow *advancedWin;
@property (assign) IBOutlet Settings *settings;
@property (assign) IBOutlet Terminal *terminal;
@property (assign) IBOutlet UltraGrid_GUIAppDelegate *app;

@end
