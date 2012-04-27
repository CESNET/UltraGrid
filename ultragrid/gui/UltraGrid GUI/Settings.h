//
//  Settings.h
//  UltraGrid GUI
//
//  Created by Martin Pulec on 11/2/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <Cocoa/Cocoa.h>



@interface Settings : NSObject {
	
	NSMutableDictionary *settings;
	NSString *filePath;
}

@property (nonatomic, readwrite) int mtu;
@property (assign) IBOutlet NSString *display;
@property (assign) IBOutlet NSString *capture;
@property (assign) IBOutlet NSString *display_details;
@property (assign) IBOutlet NSString *capture_details;
@property (assign) IBOutlet NSString *audio_cap;
@property (assign) IBOutlet NSString *audio_play;
@property (assign) IBOutlet NSString *other;

@property (assign) IBOutlet NSString * compression;

@property (nonatomic, readwrite) int JPEGQuality;

-(void) save;

@end
