//
//  DoButton.m
//  UltraGrid GUI
//
//  Created by Martin Pulec on 11/2/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import "DoButton.h"
#import "UGController.h"

@implementation DoButton

@synthesize advancedWin;
@synthesize settings;
@synthesize terminal;
@synthesize app;

- (IBAction) doPlay: (id)pId
{
	[app doPlay];
}

- (IBAction)doAdvanced:(id)pId
{
	[advancedWin makeKeyAndOrderFront: pId];
}

- (IBAction)doAdvancedOK:(id)pId
{
	[advancedWin close];
}

- (IBAction)doAdvancedSave:(id)pId
{
	[settings save];
	[advancedWin close];
}

- (IBAction)doCaptureHelp:(id)pId
{
	id controller = [[UGController alloc] init];
	NSString *arg = @"-t ";
	arg = [arg stringByAppendingString: [settings capture]];
	arg = [arg stringByAppendingString: @":help"];
	[terminal clear];
	[terminal show];
	[terminal print: [controller getOutput: arg]];
}

- (IBAction)doDisplayHelp:(id)pId
{
	id controller = [[UGController alloc] init];
	NSString *arg = @"-d ";
	arg = [arg stringByAppendingString: [settings display]];
	arg = [arg stringByAppendingString: @":help"];
	[terminal clear];
	[terminal show];
	[terminal print: [controller getOutput: arg]];
}

@end
