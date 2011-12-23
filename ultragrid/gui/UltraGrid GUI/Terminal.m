//
//  Terminal.m
//  UltraGrid GUI
//
//  Created by Martin Pulec on 11/4/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import "Terminal.h"


@implementation Terminal
@synthesize window;
@synthesize view;

-(void) show
{
	[window makeKeyAndOrderFront: self];
}

-(void) print: (NSString *) msg
{
	[[view textStorage] replaceCharactersInRange: NSMakeRange([[view textStorage] length], 
															0) withString: msg];
}

-(void) clear
{
	[[view textStorage] replaceCharactersInRange: NSMakeRange(0, [[view textStorage] length])
									  withString: @""];
}

@end
