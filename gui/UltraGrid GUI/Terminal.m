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

-(void) awakeFromNib
{
    font = [NSFont fontWithName:@"Monaco" size: 11.0];
    [font retain];

    [view setContinuousSpellCheckingEnabled:NO];
}

-(void) show
{
	[window makeKeyAndOrderFront: self];
}

-(void) print: (NSString *) msg
{
    // Get the length of the textview contents
    NSRange theEnd=NSMakeRange([[view string] length],0);
    theEnd.location+=[msg length];
    
    // Smart Scrolling
    if (NSMaxY([view visibleRect]) == NSMaxY([view bounds])) {
        // Append string to textview and scroll to bottom
        //[[textView textStorage] appendString:outputString]; - this didn't work
        [[view textStorage] replaceCharactersInRange: NSMakeRange([[view textStorage] length], 
                                                                  0) withString: msg];
        [view scrollRangeToVisible:theEnd];
    }else{
        // Append string to textview
        // [[textView textStorage] appendString:outputString];
        [[view textStorage] replaceCharactersInRange: NSMakeRange([[view textStorage] length], 
                                                                  0) withString: msg];
    }
    if(font) {
        [view setFont:font];
    }
}

-(void) clear
{
	[[view textStorage] replaceCharactersInRange: NSMakeRange(0, [[view textStorage] length])
									  withString: @""];
}

@end
