//
//  SelectFirstWordTransformer.h
//  UltraGrid GUI
//
//  Created by Martin Pulec on 11/4/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <Cocoa/Cocoa.h>


@interface SelectFirstWordTransformer : NSValueTransformer {

}

+(Class) transformedValueClass;
+(BOOL)allowsReverseTransformation;
-(id)transformedValue:(id)value;
+(id)transformedValue:(id)value;

@end
