//
//  SelectFirstWordTransformer.m
//  UltraGrid GUI
//
//  Created by Martin Pulec on 11/4/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import "SelectFirstWordTransformer.h"


@implementation SelectFirstWordTransformer
+(Class) transformedValueClass
{
	return [NSString class];
}

+(BOOL)allowsReverseTransformation
{
	return YES;
}

+(id)transformedValue:(id)value
{
	NSString *str = value;
	NSArray *array = [str componentsSeparatedByCharactersInSet:
					  [NSCharacterSet whitespaceAndNewlineCharacterSet]];
	if([array count] == 0) return nil;
	NSString *out = [[array objectAtIndex: 0] retain];
	//[array release];
	return out;
					  
}

-(id)transformedValue:(id)value
{
	return [SelectFirstWordTransformer transformedValue:value];
}

@end
