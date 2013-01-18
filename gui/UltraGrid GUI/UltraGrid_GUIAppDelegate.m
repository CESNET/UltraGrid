//
//  UltraGrid_GUIAppDelegate.m
//  UltraGrid GUI
//
//  Created by Martin Pulec on 10/31/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import "UltraGrid_GUIAppDelegate.h"
#import "UGController.h"

@implementation UltraGrid_GUIAppDelegate

@synthesize window;
@synthesize advancedWin;
@synthesize settings;
@synthesize terminal;

@synthesize remoteAddress;

@synthesize startButton;

//@synthesize displays;

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
	// Insert code here to initialize your application 
	//values = [[Settings alloc] init];
	
	execPath = [[NSBundle mainBundle] pathForAuxiliaryExecutable:@"uv"];
	NSLog(@"Executable: %s", [execPath UTF8String]);
	//[displays addObject:@"sss"];
	
	//NSLog(@"%s ", [displays lastObject]);
	controller = [[UGController alloc] init];
	[[NSNotificationCenter defaultCenter] addObserver: self
											 selector:@selector(outputReady:)
												 name:NSFileHandleDataAvailableNotification
											   object:nil ];
	
	[[NSNotificationCenter defaultCenter] addObserver: self
											 selector:@selector(terminated:)
												 name:NSTaskDidTerminateNotification
											   object:nil ];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender
{
    return YES;
}

- (void) doPlay
{
	if ([controller taskIsRunning]) {
		[controller stop];
	} else {
		[startButton setTitle: @"Stop"];
		controller = [[UGController alloc] init];
		NSMutableArray *args = [NSMutableArray arrayWithCapacity: 30];
		
		
        [args addObject: @"-m"];
		[args addObject: [NSString stringWithFormat:@"%d", settings.mtu]];
		
		if([settings.display length] > 0) {
			[args addObject: @"-d"];
			NSString *item = settings.display;
			if ([settings.display_details length] > 0) {
				item = [item stringByAppendingString: @":"];
				item = [item stringByAppendingString: settings.display_details];
			}
			[args addObject: item];
		}
		
		if([settings.capture length] > 0) {
			[args addObject: @"-t"];
			NSString *item = settings.capture;
			
			if ([settings.capture_details length] > 0) {
				item = [item stringByAppendingString: @":"];
				item = [item stringByAppendingString: settings.capture_details];
			}
			[args addObject: item];
		}
		
		if([settings.audio_play length] > 0) {
			[args addObject: @"-r"];
			[args addObject: settings.audio_play];
		}
		
		if([settings.audio_cap length] > 0) {
			[args addObject: @"-s "];
			[args addObject: settings.audio_cap];
		}
        
        if([settings.compression length] > 0) {
			[args addObject: @"-c"];
			[args addObject: [self getCompressionString]];
		}
        
		if([settings.fec length] > 0) {
            [args addObject: @"-f"];
			[args addObject: [self getFecString]];
        }
		
		if([settings.other length] > 0) {
			[args addObjectsFromArray: [settings.other componentsSeparatedByString:@" "]];
		}
        
        if([remoteAddress length] > 0) {
			[args addObject: remoteAddress];
		}
		
		[terminal clear];
		[terminal show];
        
        
		
		outputHandle = [controller getOutputHandle: args];
		[outputHandle waitForDataInBackgroundAndNotify];
		
		if(![controller taskIsRunning])
			[self terminated: [controller task]];
	}
}

-(void) outputReady: (id) sender
{
	NSData *data;
	data = [outputHandle availableData];
	if (data && [data length]) {
		NSString *string;
		string = [[NSString alloc] initWithData: data encoding: NSUTF8StringEncoding];
		[terminal print: string];
	}	
	[outputHandle waitForDataInBackgroundAndNotify];
}

-(void) terminated: (id) sender
{
	if([sender object] == [controller task])
		[startButton setTitle: @"Start UltraGrid"];
}

-(NSString *) getCompressionString
{
    NSDictionary *matching = [NSDictionary dictionaryWithObjectsAndKeys: 
                              @"none", @"none",
                              @"RTDXT:DXT1", @"DXT1",
                              @"RTDXT:DXT5", @"DXT5",
                              @"JPEG", @"JPEG",
                              @"FastDXT", @"FastDXT",
                              @"libavcodec:codec=H.264", @"H.264",
                              nil];
    
    NSString *compression = [matching objectForKey: settings.compression];
    NSString *ret;
    
    if([compression compare: @"JPEG"] == NSOrderedSame) {
        ret = [NSString stringWithFormat:@"JPEG:%d", settings.JPEGQuality];
    } else {
        ret = compression;
    }
    
    return ret;
    
}

-(NSString *) getFecString
{
    NSString *fec = settings.fec;
    NSString *ret;
    
    //NSLog(@"FEC: %@",fec);
    
    if([fec compare: @"mult"] == NSOrderedSame) {
        ret = [NSString stringWithFormat:@"mult:%d", settings.multCount];
    } else {
        ret = fec;
    }
    

    
    return ret;
}

@end
