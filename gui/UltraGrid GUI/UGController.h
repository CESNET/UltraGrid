//
//  UGController.h
//  UltraGrid GUI
//
//  Created by Martin Pulec on 11/4/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <Cocoa/Cocoa.h>


@interface UGController : NSObject {
		NSString *execPath;
	
	NSTask *task;
	NSPipe *pipe;
	NSFileHandle *file;
}

-(NSMutableArray*) getOptionsFromUG: (NSString *) param;
-(NSString *) getOutput: (id) param;
-(NSFileHandle *) getOutputHandle: (id) param;
-(BOOL) taskIsRunning;
-(void) stop;
-(NSTask*) task;

@end
