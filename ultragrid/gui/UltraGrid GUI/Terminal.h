//
//  Terminal.h
//  UltraGrid GUI
//
//  Created by Martin Pulec on 11/4/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <Cocoa/Cocoa.h>


@interface Terminal : NSObject {

}
@property (assign) IBOutlet NSWindow *window;
@property (assign) IBOutlet NSTextView *view;

-(void) show;
-(void) print: (NSString *) msg;
-(void) clear;

@end
