//
//  Settings.m
//  UltraGrid GUI
//
//  Created by Martin Pulec on 11/2/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import "Settings.h"
#import "SelectFirstWordTransformer.h"

@implementation Settings

- (id) init {
	self = [super init];
	
	filePath = [[[NSBundle mainBundle] pathForResource : @"Settings" ofType:@"plist"] retain];
	NSLog(@"%s", [filePath UTF8String]);
	
	NSData *binData = [NSData dataWithContentsOfFile:filePath];
	if(!binData)
	{
		NSLog(@"Error opening settings file!!");
	}
	NSPropertyListFormat format;
	NSString *error;
	
	NSDictionary *dict = [NSPropertyListSerialization propertyListFromData:binData
												mutabilityOption:NSPropertyListMutableContainersAndLeaves
														  format: &format errorDescription: &error];
	settings = [[NSMutableDictionary dictionaryWithCapacity: 20] retain];
	[settings addEntriesFromDictionary:dict];

	return self;
}

-(void) save
{
	NSString *error;
	
	NSData *xmlData = [NSPropertyListSerialization dataFromPropertyList: settings
																 format: NSPropertyListXMLFormat_v1_0
													   errorDescription: &error];
	NSLog(@"%s ", [error UTF8String]);
	[xmlData writeToFile:filePath atomically:YES];
	//[settings writeToFile:filePath atomically:NO];
}

-(NSString *) mtu
{
	return [settings objectForKey: @"mtu"];
}

-(void) setMtu : (NSString *) newValue;
{
	[settings setValue:newValue forKey: @"mtu"];
}

-(NSString *) display
{
	return [settings objectForKey: @"display"];
}

-(void) setDisplay : (NSString *) newValue;
{
	[settings setValue:newValue forKey: @"display"];
}

-(NSString *) display_details
{
	return [settings objectForKey: @"display_details"];
}

-(void) setDisplay_details : (NSString *) newValue;
{
	[settings setValue:newValue forKey: @"display_details"];
}

-(NSString *) capture
{
	return [settings objectForKey: @"capture"];
}

-(void) setCapture : (NSString *) newValue;
{
	[settings setValue:newValue forKey: @"capture"];
}

-(NSString *) capture_details
{
	return [settings objectForKey: @"capture_details"];
}

-(void) setCapture_details : (NSString *) newValue;
{
	[settings setValue:newValue forKey: @"capture_details"];
}

-(NSString *) audio_cap
{
	return [settings objectForKey: @"audio_cap"];
}

-(void) setAudio_cap : (NSString *) newValue;
{
	NSString *val = [SelectFirstWordTransformer transformedValue: newValue];
	
	[settings setValue: val
				forKey: @"audio_cap"];
}

-(NSString *) audio_play
{
	return [settings objectForKey: @"audio_play"];
}

-(void) setAudio_play : (NSString *) newValue;
{
	NSString *val = [SelectFirstWordTransformer transformedValue: newValue];
	
	[settings setValue: val
				forKey: @"audio_play"];
}

-(NSString *) other
{
	return [settings objectForKey: @"other"];
}

-(void) setOther : (NSString *) newValue;
{
	[settings setValue:newValue forKey: @"other"];
}

@end
