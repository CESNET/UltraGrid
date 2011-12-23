//
//  UGController.m
//  UltraGrid GUI
//
//  Created by Martin Pulec on 11/4/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import "UGController.h"


@implementation UGController
-(id) init
{
	[super init];
	execPath = [[NSBundle mainBundle] pathForAuxiliaryExecutable:@"uv"];
	return self;
}

-(NSString *) getOutput: (id) param
{
	task = [[NSTask alloc] init];
	[task setLaunchPath:execPath];
	
	NSArray *arguments;
	if([param isKindOfClass: [NSString class]]) {
		arguments = [param componentsSeparatedByString:@" "];
	} else {
		arguments = param;
	}
	
	[task setArguments: arguments];
	
	pipe = [NSPipe pipe];
	[task setStandardOutput: pipe];
	[task setStandardError: pipe];
	
	file = [pipe fileHandleForReading];
	
	[task launch];
	[task waitUntilExit];
	
	NSData *data;
	data = [file readDataToEndOfFile];
	
	NSString *string;
	string = [[NSString alloc] initWithData: data encoding: NSUTF8StringEncoding];
	return string;
	
}

-(NSFileHandle*) getOutputHandle: (id) param
{
	task = [[NSTask alloc] init];
	[task setLaunchPath:execPath];

	NSArray *arguments;
	if([param isKindOfClass: [NSString class]]) {
		arguments = [param componentsSeparatedByString:@" "];
	} else {
		arguments = param;
	}
	
	[task setArguments: arguments];
	
	pipe = [NSPipe pipe];
	[task setStandardOutput: pipe];
	[task setStandardError: pipe];
	
	file = [pipe fileHandleForReading];
	[task launch];
	
	return file;
}

-(NSMutableArray*) getOptionsFromUG: (NSString *) param
{
	NSString *string = [self getOutput: param];
	
	NSArray *array = [string componentsSeparatedByCharactersInSet:
					  [NSCharacterSet newlineCharacterSet]];
	NSMutableArray *ret = [NSMutableArray arrayWithCapacity: 20];
	for(NSString *item in array)
	{
		if([item length] > 0 &&
		   [[NSCharacterSet whitespaceCharacterSet] characterIsMember: [item characterAtIndex:0]]) {
			NSString *out = [item stringByTrimmingCharactersInSet:
							 [NSCharacterSet whitespaceCharacterSet]];
			NSLog(@"retuned: %@", out);
			[ret addObject: out];
		}
	}
	
	return ret;
}

-(BOOL) taskIsRunning
{
	return [task isRunning];
}

-(void) stop
{
	[task interrupt];sleep(1);;
	[task terminate];
}

-(NSTask*) task
{
	return task;
}

@end
