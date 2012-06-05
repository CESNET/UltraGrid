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

-(int) mtu
{
    NSString *MTUStr = [settings objectForKey: @"mtu"];
    int ret = 1500;
    if(MTUStr)
        ret = [MTUStr integerValue];
    
    return ret;
}

-(void) setMtu:(int)newVal
{
    if (newVal < 500) {
        newVal = 500;
    }
    if(newVal > 9000) {
        newVal = 9000;
    }
    NSString *MTUStr = [NSString stringWithFormat: @"%d", newVal];
    [settings setValue:MTUStr forKey: @"mtu"];
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

-(NSString *) compression
{
	NSString *ret = [settings objectForKey: @"compression"];
    
    if (ret == nil) {
        ret = @"none";
    }
    
    return ret;
}

-(void) setCompression : (NSString *) newValue;
{
	[settings setValue:newValue forKey: @"compression"];
    //NSLog(@"%@", newValue);
}

-(int) JPEGQuality
{
    NSString *JPEGQualityStr = [settings objectForKey: @"jpeg_quality"];
    int ret;
    if(JPEGQualityStr)
        ret = [JPEGQualityStr integerValue];
    else
        ret = 80;

    return ret;
}

-(void) setJPEGQuality: (int) newVal
{
    if (newVal < 0) {
        newVal = 0;
    }
    if(newVal > 100) {
        newVal = 100;
    }
    NSString *JPEGQuality = [NSString stringWithFormat: @"%d", newVal];
    [settings setValue:JPEGQuality forKey: @"jpeg_quality"];
}

-(NSString *) fec
{
	NSString *ret = [settings objectForKey: @"fec"];
    
    if(ret == nil) {
        ret = @"none";
    }

    return ret;
}

-(void) setFec: (NSString *) newValue;
{
	[settings setValue:newValue forKey: @"fec"];
    //NSLog(@"%@", newValue);
}


-(int) multCount
{
    int ret = 2;
    NSString *multCountStr = [settings objectForKey: @"mult_count"];
    if(multCountStr) {
        ret = [multCountStr integerValue];
    }
    
    return ret;
}

-(void) setMultCount:(int)multCount
{
    if (multCount < 1) {
        multCount = 1;
    }
    if(multCount > 10) {
        multCount = 10;
    }
    NSString *multCountStr = [NSString stringWithFormat: @"%d", multCount];
    [settings setValue:multCountStr forKey: @"mult_count"];
}

@end
